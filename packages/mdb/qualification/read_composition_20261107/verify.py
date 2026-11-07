"""Offline verification for the 2026-11-07 read-composition assay.

The worker in this qualification deliberately writes a fairly small report on
top of a full Bench session tree.  This module checks that report without
loading a model: the public Bench verifier checks the session/CAS closure and
the checks below bind the frozen read program, the native capture payloads,
the value-only forecasts, and the continuation records to those sessions.

The verifier reports mechanics and instrument validity separately from the
measured effects.  It never turns an effect size, answer accuracy, or a
semantic-CoT flag into a pass/fail claim.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from mdb.bench_circuit_discovery_verify import (
    _close_number,
    _close_structure,
    _cut,
    _tensor_digest,
    _verify_receipt,
)
from mdb.bench_hidden_memory import verify_write_act
from mdb.bench_verify import verify as verify_bench
from mdb.circuit_runtime import DirectoryArtifactStore
from mdb.model_os_contracts import content_digest
from mdb.read_program import FrozenReadProgram

REPORT_SCHEMA = "mdb-read-composition-report-v1"
VERIFY_SCHEMA = "mdb-read-composition-verification-v1"
PREDICTION_SCHEMA = "mdb-two-read-composition-prediction-v1"
_CUT_ABI_BITS = 64


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, path: str = "value") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite report number at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def _dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} is malformed")
    return value


def _ref(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} reference is missing")
    return value


def _get(store: DirectoryArtifactStore, reference: str, fallback=None):
    try:
        return store.get(reference)
    except Exception:
        if fallback is None:
            raise
        return fallback.get(reference)


def _load_case_stores(case_root: Path):
    artifacts = DirectoryArtifactStore(case_root / "artifacts")
    vm_artifacts = DirectoryArtifactStore(case_root / "vm" / "artifacts")
    selected = json.loads((case_root / "selected.json").read_text())
    manifest_ref = _ref(selected.get("manifest"), "session manifest")
    manifest = _dict(artifacts.get(manifest_ref), "session manifest")
    return artifacts, vm_artifacts, manifest_ref, manifest


def _verify_bench_read_only(case_root: Path) -> dict[str, Any]:
    """Run the public verifier on a disposable copy of one inventoried tree.

    The public API is the authoritative session check, but its failure path
    may persist a diagnostic audit.  Keeping that possible side effect out of
    the recovered result tree makes this qualification audit observational.
    """

    with tempfile.TemporaryDirectory(prefix="mdb-read-composition-verify-") as directory:
        copy_root = Path(directory) / case_root.name
        shutil.copytree(case_root, copy_root, symlinks=False)
        return verify_bench(copy_root)


def _load_cut(reference: str, artifacts, manifest, *, contract: str | None = None):
    return _cut(reference, artifacts, manifest, numerical_contract=contract)


def _tensor_equal(left, right) -> bool:
    import torch

    return tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype and bool(torch.equal(left, right))


def _error(actual, expected) -> dict[str, Any]:
    """Match worker.error while keeping reductions offline and deterministic."""

    delta = actual.float() - expected.float()
    expected_norm = float(expected.float().norm())
    delta_norm = float(delta.norm())
    return {
        "l2": delta_norm,
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "relative_l2": delta_norm / expected_norm if expected_norm else None,
        "exact": _tensor_digest(actual) == _tensor_digest(expected),
    }


def _payload(reference: Any, vm_artifacts, label: str):
    """Hydrate through the read-only payload audit path.

    ``PayloadStore.hydrate`` deliberately publishes a native-consumer receipt
    and is therefore not appropriate for an audit of an inventoried result.
    ``audit`` verifies the same bytes and geometry without mutating the CAS.
    """

    import torch

    from mdb.vm_payload_cache import PayloadRef, PayloadStore

    try:
        payload = PayloadRef.from_dict(reference)
        store = PayloadStore(vm_artifacts.root.parent / "radix" / "cas", read_only=True)
        raw = store.audit(payload).payload
        if sha256(raw).hexdigest() != payload.sha256:
            raise ValueError(f"{label} payload digest mismatch")
        dtype = getattr(torch, payload.dtype.removeprefix("torch."))
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(payload.shape)).clone()
    except Exception as error:
        raise ValueError(f"{label} tensor payload is malformed") from error
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{label} tensor payload is non-finite")
    return tensor, payload


def _artifact_digest(reference: str, artifact: Mapping[str, Any], label: str) -> None:
    if not reference.startswith("cas://sha256/"):
        raise ValueError(f"{label} is not a CAS reference")
    if content_digest(artifact) != reference.removeprefix("cas://sha256/"):
        raise ValueError(f"{label} digest binding changed")


def _close_comparison(actual: Any, expected: Any, *, label: str) -> None:
    """Compare worker future summaries with the calibrated large-vocab norm tolerance."""

    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} fields changed")
        for key in expected:
            if key in {"l2", "logit_l2"}:
                _close_number(actual[key], expected[key], label=f"{label}.{key}", relative_tolerance=1e-4)
            else:
                _close_comparison(actual[key], expected[key], label=f"{label}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length changed")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _close_comparison(left, right, label=f"{label}[{index}]")
    elif isinstance(actual, float) or isinstance(expected, float):
        _close_number(actual, expected, label=label)
    elif actual != expected:
        raise ValueError(f"{label} changed")


def _check_program(case: Mapping[str, Any], program: FrozenReadProgram) -> FrozenReadProgram:
    try:
        case_program = FrozenReadProgram.from_dict(case["program"])
    except Exception as error:
        raise ValueError("case FrozenReadProgram is invalid") from error
    if case_program.to_dict() != program.to_dict():
        raise ValueError("case FrozenReadProgram differs from report declaration")
    return case_program


def _span(value: Any, label: str, *, limit: int | None = None) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} span is malformed")
    start, stop = (_int(value[0], f"{label} start"), _int(value[1], f"{label} stop"))
    if start >= stop or (limit is not None and stop > limit):
        raise ValueError(f"{label} span is out of bounds")
    return start, stop


def _site(
    site: Mapping[str, Any],
    port_name: str,
    program: FrozenReadProgram,
    parent: Mapping[str, Any],
    *,
    expected_span: tuple[int, int] | None = None,
    copula_token_id: int | None = None,
) -> tuple[int, int]:
    if not isinstance(site, Mapping):
        raise ValueError(f"resolved {port_name} site is malformed")
    port = program.port(port_name)
    for key, expected in (
        ("port", port_name),
        ("layer", port.layer),
        ("kv_head", port.kv_head),
        ("role", port.row_role),
        ("provenance", port.provenance),
    ):
        if site.get(key) != expected:
            raise ValueError(f"resolved {port_name} {key} binding changed")
    row = _int(site.get("row"), f"resolved {port_name} row")
    if site.get("source_row") != row:
        raise ValueError(f"resolved {port_name} source-row alias changed")
    tokens = parent["token_ids"]
    if row >= len(tokens) or site.get("source_token_id") != tokens[row]:
        raise ValueError(f"resolved {port_name} source-token binding changed")
    lineage = _dict(site.get("lineage"), f"resolved {port_name} lineage")
    if lineage.get("provenance") != port.provenance or lineage.get("role") != port.row_role:
        raise ValueError(f"resolved {port_name} lineage role changed")
    span = _span(lineage.get("span"), f"resolved {port_name} lineage", limit=len(tokens))
    if expected_span is not None and span != expected_span:
        raise ValueError(f"resolved {port_name} lineage span changed")
    if port_name == "A":
        if lineage.get("source") != "preceding_answer_span" or lineage.get("occurrence") != "first-generated":
            raise ValueError("resolved A lineage occurrence changed")
        if row != span[0]:
            raise ValueError("resolved A is not the first row in its supplied answer span")
    else:
        if lineage.get("source") != "followup_span" or lineage.get("occurrence") != "last-copula-in-span":
            raise ValueError("resolved B lineage occurrence changed")
        if copula_token_id is None or lineage.get("copula_token_id") != copula_token_id:
            raise ValueError("resolved B copula binding changed")
        rows = [index for index in range(span[0], span[1]) if tokens[index] == copula_token_id]
        if not rows or row != rows[-1]:
            raise ValueError("resolved B is not the last in-span copula row")
    return span


def _check_sites(
    sites: Mapping[str, Any],
    program: FrozenReadProgram,
    parent: Mapping[str, Any],
    *,
    answer_span: tuple[int, int],
    copula_token_id: int,
) -> dict[str, tuple[int, int]]:
    if not isinstance(sites, Mapping):
        raise ValueError("resolved site map is missing")
    if not {"A", "B", "wrong"} <= set(sites):
        raise ValueError("resolved site map is incomplete")
    a_span = _site(sites["A"], "A", program, parent, expected_span=answer_span)
    b_span = _site(sites["B"], "B", program, parent, copula_token_id=copula_token_id)
    if a_span[1] > b_span[0]:
        raise ValueError("resolved A/B lineage spans overlap or are reversed")
    wrong = _dict(sites["wrong"], "wrong-row control")
    if wrong.get("role") != "wrong-row-control" or wrong.get("layer") != program.port("A").layer:
        raise ValueError("wrong-row control declaration changed")
    wrong_row = _int(wrong.get("source_row"), "wrong-row source row")
    if wrong_row >= len(parent["token_ids"]) or wrong.get("source_token_id") != parent["token_ids"][wrong_row]:
        raise ValueError("wrong-row source binding changed")
    return {"A": a_span, "B": b_span}


def _verify_arm(
    arm: Mapping[str, Any],
    *,
    label: str,
    parent_ref: str,
    parent: Mapping[str, Any],
    artifacts,
    vm_artifacts,
    manifest: Mapping[str, Any],
    contract: str,
    cache: dict[str, tuple[Any, Any]],
) -> Mapping[str, Any]:
    arm = _dict(arm, f"arm {label}")
    if arm.get("name") != label or arm.get("parent_cut") != parent_ref:
        raise ValueError(f"arm {label} parent/name binding changed")
    trajectory = _list(arm.get("trajectory"), f"arm {label} trajectory")
    if len(trajectory) != 3:
        raise ValueError(f"arm {label} must have three native steps")
    previous_ref = parent_ref
    previous = parent
    for index, row_value in enumerate(trajectory):
        row = _dict(row_value, f"arm {label} step")
        if row.get("offset") != index:
            raise ValueError(f"arm {label} offset changed")
        token = _int(row.get("consumed_token_id"), f"arm {label} consumed token")
        if row.get("authority") != ("caller-supplied" if index == 0 else "native-greedy"):
            raise ValueError(f"arm {label} input authority changed")
        cut_ref = _ref(row.get("cut"), f"arm {label} cut")
        cut = _load_cut(cut_ref, artifacts, manifest, contract=contract)
        committed_parent = cut.get("parent_cut")
        if index == 0:
            staged = _load_cut(_ref(committed_parent, f"arm {label} staged cut"), artifacts, manifest, contract=contract)
            # A memory Act may be committed between the saved bundle parent
            # and stage_input (and reinstall has two additional copy Acts).
            # Follow the immutable cut ancestry rather than requiring the
            # provisional cut to point directly at the bundle parent.
            ancestor = staged.get("parent_cut")
            ancestry = set()
            while ancestor is not None and ancestor not in ancestry:
                ancestry.add(ancestor)
                if ancestor == previous_ref:
                    break
                ancestor_cut = _load_cut(ancestor, artifacts, manifest, contract=contract)
                ancestor = ancestor_cut.get("parent_cut")
            if (
                previous_ref not in ancestry
                or staged.get("token_ids") != previous["token_ids"]
                or staged.get("pending") != token
                or staged.get("label") != "supplied-input-embedding"
            ):
                raise ValueError(f"arm {label} supplied-input staging changed")
        elif committed_parent != previous_ref:
            raise ValueError(f"arm {label} step parent changed")
        expected_tokens = list(previous["token_ids"]) + [token]
        if cut.get("token_ids") != expected_tokens or row.get("consumed_token_id") != expected_tokens[-1]:
            raise ValueError(f"arm {label} token history changed")
        _verify_receipt(row.get("receipt_ref"), artifacts, vm_artifacts, row=row)
        if cut.get("pending") != row.get("next_token_id"):
            raise ValueError(f"arm {label} pending-token binding changed")
        logits_ref = _dict(cut.get("logits"), f"arm {label} logits")
        logits, logits_payload = _payload(logits_ref, vm_artifacts, f"arm {label} logits")
        if row.get("logits_sha256") != logits_payload.sha256:
            raise ValueError(f"arm {label} logits digest binding changed")
        import torch

        expected_next = int(torch.argmax(logits.float()[0]).item())
        if row.get("next_token_id") != expected_next:
            raise ValueError(f"arm {label} next-token readout changed")
        previous_ref, previous = cut_ref, cut
    if arm.get("end_cut") != previous_ref:
        raise ValueError(f"arm {label} end cut changed")
    closure = arm.get("closure_sha256")
    if not _sha(closure):
        raise ValueError(f"arm {label} closure digest is missing")
    return arm


def _verify_acts(
    arms: Mapping[str, Any],
    *,
    sites: Mapping[str, Any],
    parent_ref: str,
    artifacts,
    vm_artifacts,
) -> None:
    for label, arm_value in arms.items():
        arm = _dict(arm_value, f"arm {label}")
        acts = arm.get("acts", [])
        if not isinstance(acts, list):
            raise ValueError(f"arm {label} Acts are malformed")
        previous_act_cut = parent_ref
        for item_value in acts:
            item = _dict(item_value, f"arm {label} Act item")
            port = item.get("port")
            if port not in sites:
                raise ValueError(f"arm {label} Act port is unknown")
            act = _dict(item.get("act"), f"arm {label} Act")
            verify_write_act(act, artifacts, vm_artifacts)
            if act.get("parent_cut") != previous_act_cut or act.get("donor_cut") != previous_act_cut:
                raise ValueError(f"arm {label} Act parent/donor changed")
            site = sites[port]
            expected_selector = {
                "layers": [site["layer"]],
                "components": ["value"],
                "row_start": site["source_row"],
                "row_stop": site["source_row"] + 1,
            }
            if act.get("selector") != expected_selector or act.get("mode") != "replace":
                raise ValueError(f"arm {label} {port} Act selector changed")
            previous_act_cut = _ref(act.get("cut_id"), f"arm {label} Act cut")


def _capture_tensor(
    arm: Mapping[str, Any],
    name: str,
    vm_artifacts,
    *,
    required: bool = True,
):
    capture = _dict(arm.get("capture"), f"arm {arm.get('name')} capture")
    reference = capture.get(name)
    if reference is None and not required:
        return None
    return _payload(reference, vm_artifacts, f"arm {arm.get('name')} {name}")


def _verify_controls(arms: Mapping[str, Any]) -> None:
    def signature(arm):
        return [(row["consumed_token_id"], row["logits_sha256"]) for row in arm["trajectory"]]

    zero = _dict(arms["00"], "00 arm")
    for name in ("sham", "reinstall"):
        arm = _dict(arms[name], name)
        if signature(arm) != signature(zero) or arm.get("closure_sha256") != zero.get("closure_sha256"):
            raise ValueError(f"{name} observer/reinstall continuation closure changed")


def _verify_prediction(
    bundle: Mapping[str, Any],
    *,
    arms: Mapping[str, Any],
    artifacts,
    vm_artifacts,
    cache: dict[str, tuple[Any, Any]],
) -> dict[str, dict[str, float | bool | None]]:
    prediction_ref = _ref(bundle.get("prediction_ref"), "prediction")
    prediction = _dict(artifacts.get(prediction_ref), "prediction artifact")
    _artifact_digest(prediction_ref, prediction, "prediction artifact")
    if prediction.get("schema") != PREDICTION_SCHEMA or prediction != bundle.get("prediction_record"):
        raise ValueError("prediction artifact/report binding changed")
    if prediction.get("parent_cut") != bundle.get("parent_cut"):
        raise ValueError("prediction parent binding changed")
    if prediction.get("program_status") != "frozen-law; supplied-role-constructors; no fitting":
        raise ValueError("prediction program status changed")
    predictions = _dict(prediction.get("predictions"), "prediction records")
    if set(predictions) != {"conditional", "stale"}:
        raise ValueError("prediction records are incomplete")

    a_capture = _dict(arms["A"].get("capture"), "A capture")
    zero_capture = _dict(arms["00"].get("capture"), "00 capture")
    outputs: dict[str, dict[str, float | bool | None]] = {}
    for name in ("conditional", "stale"):
        record = _dict(predictions[name], f"{name} prediction")
        expected_contribution_ref = a_capture["contribution_B"] if name == "conditional" else zero_capture["contribution_B"]
        for key, expected in (
            ("input22_ref", a_capture["input22"]),
            ("baseline_attention_ref", a_capture["attention22"]),
            ("contribution_ref", expected_contribution_ref),
        ):
            if record.get(key) != expected:
                raise ValueError(f"{name} prediction {key} binding changed")
        if record.get("native_attention_output_assisted") is not True or record.get("injected_replay_native_attention_omitted") is not False:
            raise ValueError(f"{name} prediction native-assistance declaration changed")
        if record.get("dense_mlp_linear_calls") != 3:
            raise ValueError(f"{name} prediction dense-call declaration changed")
        baseline, baseline_payload = _payload(a_capture["attention22"], vm_artifacts, f"{name} baseline attention")
        contribution, _ = _payload(expected_contribution_ref, vm_artifacts, f"{name} value contribution")
        predicted, predicted_payload = _payload(record.get("attention_ref"), vm_artifacts, f"{name} predicted attention")
        consumed, consumed_payload = _payload(record.get("consumed_attention_ref"), vm_artifacts, f"{name} consumed attention")
        import torch

        try:
            expected = baseline.float() - contribution.float()
        except RuntimeError as error:
            raise ValueError(f"{name} attention/contribution geometry changed") from error
        if tuple(predicted.shape) != tuple(expected.shape) or not bool(torch.equal(predicted.float(), expected.float())):
            raise ValueError(f"{name} attention prediction formula changed")
        if consumed.dtype != baseline.dtype or not _tensor_equal(consumed, predicted.to(dtype=baseline.dtype)):
            raise ValueError(f"{name} BF16/native attention cast changed")
        injected_attention, _ = _capture_tensor(arms[name], "attention22", vm_artifacts)
        if not _tensor_equal(injected_attention, consumed):
            raise ValueError(f"{name} injected attention does not match sealed cast")
        predicted_mlp, _ = _payload(record.get("mlp_ref"), vm_artifacts, f"{name} predicted MLP")
        predicted_exit, _ = _payload(record.get("exit_ref"), vm_artifacts, f"{name} predicted exit")
        injected_mlp, _ = _capture_tensor(arms[name], "mlp22", vm_artifacts)
        injected_exit, _ = _capture_tensor(arms[name], "exit22", vm_artifacts)
        if not _tensor_equal(predicted_mlp, injected_mlp) or not _tensor_equal(predicted_exit, injected_exit):
            raise ValueError(f"{name} predicted MLP/exit bytes differ from injected capture")
        outputs[name] = {
            "attention_formula_exact": True,
            "bf16_cast_exact": True,
            "mlp_injected_exact": True,
            "exit_injected_exact": True,
            "predicted_attention_bytes": predicted_payload.sha256,
            "consumed_attention_bytes": consumed_payload.sha256,
            "baseline_attention_bytes": baseline_payload.sha256,
        }
    return outputs


def _verify_local_errors(
    bundle: Mapping[str, Any],
    *,
    arms: Mapping[str, Any],
    predictions: Mapping[str, Any],
    vm_artifacts,
) -> dict[str, Any]:
    reported = _dict(bundle.get("local_errors"), "local errors")
    result: dict[str, Any] = {}
    for name in ("conditional", "stale"):
        expected = _dict(reported.get(name), f"{name} local errors")
        prediction = _dict(predictions[name], f"{name} prediction")
        actuals = {
            "attention": _capture_tensor(arms["AB"], "attention22", vm_artifacts)[0],
            "mlp": _capture_tensor(arms["AB"], "mlp22", vm_artifacts)[0],
            "exit": _capture_tensor(arms["AB"], "exit22", vm_artifacts)[0],
        }
        predicted = {
            "attention": _payload(prediction["attention_ref"], vm_artifacts, f"{name} attention")[0],
            "mlp": _payload(prediction["mlp_ref"], vm_artifacts, f"{name} MLP")[0],
            "exit": _payload(prediction["exit_ref"], vm_artifacts, f"{name} exit")[0],
        }
        result[name] = {}
        for key in ("attention", "mlp", "exit"):
            # Worker ordering is error(predicted, native-AB), so the
            # relative norm is the native consumer's norm as well.
            calculated = _error(predicted[key], actuals[key])
            supplied = _dict(expected.get(key), f"{name} {key} local error")
            _close_structure(supplied, calculated, label=f"{name} {key} local error")
            result[name][key] = calculated
        for key, value in (
            ("injected_mlp_exact", _tensor_equal(_capture_tensor(arms[name], "mlp22", vm_artifacts)[0], predicted["mlp"])),
            ("injected_exit_exact", _tensor_equal(_capture_tensor(arms[name], "exit22", vm_artifacts)[0], predicted["exit"])),
        ):
            if expected.get(key) is not value:
                raise ValueError(f"{name} {key} changed")
            result[name][key] = value
    return result


def _verify_route_and_interaction(
    bundle: Mapping[str, Any],
    *,
    arms: Mapping[str, Any],
    vm_artifacts,
) -> dict[str, Any]:
    a_q = _capture_tensor(arms["A"], "q22", vm_artifacts)[0]
    ab_q = _capture_tensor(arms["AB"], "q22", vm_artifacts)[0]
    a_k = _capture_tensor(arms["A"], "k22", vm_artifacts)[0]
    ab_k = _capture_tensor(arms["AB"], "k22", vm_artifacts)[0]
    a_v = _capture_tensor(arms["A"], "current_v22", vm_artifacts)[0]
    ab_v = _capture_tensor(arms["AB"], "current_v22", vm_artifacts)[0]
    a_input = _capture_tensor(arms["A"], "input22", vm_artifacts)[0]
    ab_input = _capture_tensor(arms["AB"], "input22", vm_artifacts)[0]
    a_source = _capture_tensor(arms["A"], "source_B", vm_artifacts)[0]
    zero_source = _capture_tensor(arms["00"], "source_B", vm_artifacts)[0]
    route_exact = all((_tensor_equal(a_q, ab_q), _tensor_equal(a_k, ab_k), _tensor_equal(a_v, ab_v)))
    if bundle.get("A_vs_AB_query_keys_current_value_exact") is not route_exact:
        raise ValueError("A/AB query-key-current-value declaration changed")
    if bundle.get("A_vs_AB_input22_exact") is not _tensor_equal(a_input, ab_input):
        raise ValueError("A/AB input declaration changed")
    if bundle.get("A_vs_00_source_B_exact") is not _tensor_equal(a_source, zero_source):
        raise ValueError("A/00 source-B declaration changed")
    if bundle.get("old_keys22_unchanged") is not _tensor_equal(a_k[:, :, :-1], ab_k[:, :, :-1]):
        raise ValueError("old-key locality declaration changed")
    metrics = {
        "query22_change": _error(a_q, _capture_tensor(arms["00"], "q22", vm_artifacts)[0]),
        "current_key22_change": _error(a_k[:, :, -1:], _capture_tensor(arms["00"], "k22", vm_artifacts)[0][:, :, -1:]),
        "current_value22_change": _error(a_v, _capture_tensor(arms["00"], "current_v22", vm_artifacts)[0]),
    }
    for key, expected in metrics.items():
        _close_structure(_dict(bundle.get(key), key), expected, label=key)
    # The four-way exit interaction is persisted as a payload and as a norm.
    interaction = (
        _capture_tensor(arms["AB"], "exit22", vm_artifacts)[0].float()
        - _capture_tensor(arms["A"], "exit22", vm_artifacts)[0].float()
        - _capture_tensor(arms["B"], "exit22", vm_artifacts)[0].float()
        + _capture_tensor(arms["00"], "exit22", vm_artifacts)[0].float()
    )
    interaction_ref = bundle.get("layer22_exit_interaction_ref")
    persisted, persisted_payload = _payload(interaction_ref, vm_artifacts, "exit interaction")
    if not _tensor_equal(persisted, interaction):
        raise ValueError("layer-22 exit interaction payload changed")
    _close_number(bundle.get("layer22_exit_interaction_l2"), float(interaction.norm()), label="exit interaction l2")
    return {"route_exact": route_exact, **metrics, "interaction_l2": float(interaction.norm()), "interaction_ref": persisted_payload.sha256}


def _compare_arms(candidate: Mapping[str, Any], reference: Mapping[str, Any], *, artifacts, vm_artifacts) -> dict[str, Any]:
    rows = []
    candidate_rows = _list(candidate.get("trajectory"), "candidate trajectory")
    reference_rows = _list(reference.get("trajectory"), "reference trajectory")
    if len(candidate_rows) != len(reference_rows):
        raise ValueError("future comparison trajectory lengths differ")
    for left_value, right_value in zip(candidate_rows, reference_rows, strict=True):
        left, right = _dict(left_value, "candidate trajectory row"), _dict(right_value, "reference trajectory row")
        left_cut = artifacts.get(left["cut"])
        right_cut = artifacts.get(right["cut"])
        left_logits, _ = _payload(left_cut["logits"], vm_artifacts, "candidate future logits")
        right_logits, _ = _payload(right_cut["logits"], vm_artifacts, "reference future logits")
        calculated = _error(left_logits, right_logits)
        rows.append({
            "offset": left["offset"],
            "same_consumed": left["consumed_token_id"] == right["consumed_token_id"],
            "same_next": left["next_token_id"] == right["next_token_id"],
            "target_margin_change": left["target_margin"] - right["target_margin"],
            **calculated,
        })
    return rows


def _verify_future_comparisons(bundle: Mapping[str, Any], *, arms: Mapping[str, Any], artifacts, vm_artifacts) -> dict[str, Any]:
    supplied = _dict(bundle.get("future_comparisons"), "future comparisons")
    result = {}
    for name in ("conditional", "stale"):
        calculated = _compare_arms(arms[name], arms["AB"], artifacts=artifacts, vm_artifacts=vm_artifacts)
        # The worker persists a compact list rather than the discovery verifier's
        # richer structure.  Compare derived norms with the 100-ppm logit norm
        # tolerance used by the public verifier.
        _close_comparison(_list(supplied.get(name), f"{name} future comparison"), calculated,
                          label=f"{name} future comparison")
        result[name] = calculated
    return result


def _verify_semantic_metrics(case_result: Mapping[str, Any], *, artifacts, vm_artifacts) -> dict[str, Any]:
    first = _dict(case_result.get("first"), "first bundle")
    target_ids = first.get("semantic_target_ids")
    if target_ids is None:
        return {"status": "not-recorded"}
    if not isinstance(target_ids, list) or len(target_ids) != 2 or any(type(token) is not int or token < 0 for token in target_ids):
        raise ValueError("semantic target IDs are malformed")
    initial = _list(case_result.get("initial_native"), "initial native steps")
    if len(initial) != 2:
        raise ValueError("initial native trajectory does not contain two steps")
    expected = _int(case_result.get("initial_expected_id"), "initial expected token")
    correct = initial[0].get("token_id") == expected
    if case_result.get("initial_answer_correct") is not correct:
        raise ValueError("initial semantic accuracy binding changed")
    margins = []
    for index, row_value in enumerate(initial):
        row = _dict(row_value, "initial native row")
        cut = _dict(artifacts.get(_ref(row.get("cut"), "initial cut")), "initial cut")
        logits, _ = _payload(cut.get("logits"), vm_artifacts, f"initial logits {index}")
        against = target_ids[1] if target_ids[0] == expected else target_ids[0]
        margins.append({
            "offset": index,
            "consumed_token_id": row.get("token_id"),
            "expected_token_id": expected,
            "correct": row.get("token_id") == expected,
            "target_margin": float(logits.float()[0, expected] - logits.float()[0, against]),
        })
    return {"status": "measured", "target_ids": target_ids, "expected_token_id": expected,
            "first_token_correct": correct, "margins": margins}


def _verify_prediction_events(case_result: Mapping[str, Any], bundles: list[Mapping[str, Any]]) -> dict[str, Any]:
    events = _list(case_result.get("prediction_events"), "prediction events")
    positions = []
    for bundle in bundles:
        prediction_ref = _ref(bundle.get("prediction_ref"), "bundle prediction")
        seals = [(index, event) for index, event in enumerate(events)
                 if isinstance(event, dict) and event.get("kind") == "prediction-sealed" and event.get("ref") == prediction_ref]
        starts = [(index, event) for index, event in enumerate(events)
                  if isinstance(event, dict) and event.get("kind") == "native-AB-started" and event.get("prediction_ref") == prediction_ref]
        if len(seals) != 1 or len(starts) != 1:
            raise ValueError("prediction seal/AB event pair is incomplete")
        seal_index, seal = seals[0]
        start_index, start = starts[0]
        if seal_index >= start_index or seal.get("physical_steps") is None or start.get("physical_steps") is None:
            raise ValueError("prediction was not sealed before native AB")
        if seal["physical_steps"] > start["physical_steps"]:
            raise ValueError("prediction physical step ordering changed")
        positions.append({"prediction_ref": prediction_ref, "seal_index": seal_index, "AB_index": start_index,
                          "seal_steps": seal["physical_steps"], "AB_steps": start["physical_steps"]})
    return {"status": "valid", "pairs": positions}


def _verify_recurrent(case_result: Mapping[str, Any], *, program: FrozenReadProgram, artifacts, vm_artifacts, manifest, contract: str, cache: dict[str, Any]) -> dict[str, Any]:
    first = _dict(case_result["first"], "first bundle")
    policies = _dict(case_result.get("recurrent_policies"), "recurrent policies")
    if set(policies) != {"00", "A", "B", "AB"}:
        raise ValueError("recurrent policy set changed")
    first_arms = _dict(first["arms"], "first arms")
    checked = {}
    for policy_name, policy_value in policies.items():
        policy = _dict(policy_value, f"recurrent policy {policy_name}")
        world = _dict(first_arms[policy_name], f"first world {policy_name}")
        if policy.get("first_world_cut") != world.get("end_cut"):
            raise ValueError(f"recurrent {policy_name} first-world lineage changed")
        world_cut = _load_cut(world["end_cut"], artifacts, manifest, contract=contract)
        answer_span = _span(policy.get("preceding_answer_span"), f"recurrent {policy_name} answer", limit=len(world_cut["token_ids"]))
        sites = _dict(policy.get("resolved_sites"), f"recurrent {policy_name} sites")
        _check_sites(sites, program, _dict(artifacts.get(_ref(policy["outcome"].get("parent_cut"), "recurrent parent cut")), "recurrent parent cut"), answer_span=answer_span, copula_token_id=case_result["copula_token_id"])
        outcome = _dict(policy.get("outcome"), f"recurrent {policy_name} outcome")
        outcome_parent_ref = _ref(outcome.get("parent_cut"), f"recurrent {policy_name} outcome parent")
        parent = _load_cut(outcome_parent_ref, artifacts, manifest, contract=contract)
        if sites["A"].get("lineage", {}).get("span") != list(answer_span):
            raise ValueError(f"recurrent {policy_name} A lineage changed")
        _verify_arm(outcome, label=outcome.get("name", f"repeat-{policy_name}"), parent_ref=outcome_parent_ref,
                    parent=parent, artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest,
                    contract=contract, cache=cache)
        checked[policy_name] = {"first_world_cut": world["end_cut"], "parent_cut": outcome_parent_ref,
                                "answer_span": list(answer_span)}
    return checked


def verify_case(case_result: Mapping[str, Any], case_root: str | Path, *, program: FrozenReadProgram) -> dict[str, Any]:
    """Verify one recovered case and its complete Bench session tree."""

    case_root = Path(case_root)
    if not case_root.is_dir():
        raise ValueError(f"case session directory does not exist: {case_root}")
    bench_summary = _verify_bench_read_only(case_root)
    artifacts, vm_artifacts, manifest_ref, manifest = _load_case_stores(case_root)
    if case_result.get("model_identity") != manifest.get("identity"):
        raise ValueError("case model identity differs from session manifest")
    if case_result.get("numerical_contract") != manifest.get("capabilities", {}).get("numerical_contract"):
        raise ValueError("case native numerical profile differs from session manifest")
    if manifest.get("attention_backend") != "native-eager":
        raise ValueError("read-composition assay is not native-eager")
    _check_program(case_result, program)
    copula = _int(case_result.get("copula_token_id"), "copula token")
    first = _dict(case_result.get("first"), "first bundle")
    parent_ref = _ref(first.get("parent_cut"), "first bundle parent")
    parent = _load_cut(parent_ref, artifacts, manifest)
    contract = parent["numerical_contract"]
    if not _sha(contract) or len(contract) != _CUT_ABI_BITS:
        raise ValueError("first parent cut ABI fingerprint is malformed")
    answer_span = _span(case_result.get("initial_answer_span"), "initial answer", limit=len(parent["token_ids"]))
    program_sites = _check_sites(first.get("sites"), program, parent, answer_span=answer_span, copula_token_id=copula)
    arms = _dict(first.get("arms"), "first arms")
    required_arms = {"00", "sham", "A", "B", "wrong-row", "reinstall", "conditional", "stale", "AB"}
    if set(arms) != required_arms:
        raise ValueError("first arm set changed")
    cache: dict[str, tuple[Any, Any]] = {}
    arm_summary = {}
    for name in sorted(arms):
        arm_summary[name] = _verify_arm(arms[name], label=name, parent_ref=parent_ref, parent=parent,
                                        artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest,
                                        contract=contract, cache=cache)
    _verify_acts(arms, sites=first["sites"], parent_ref=parent_ref, artifacts=artifacts, vm_artifacts=vm_artifacts)
    _verify_controls(arms)
    prediction_summary = _verify_prediction(first, arms=arms, artifacts=artifacts, vm_artifacts=vm_artifacts, cache=cache)
    error_summary = _verify_local_errors(first, arms=arms, predictions=_dict(first["prediction_record"]["predictions"], "prediction records"), vm_artifacts=vm_artifacts)
    route_summary = _verify_route_and_interaction(first, arms=arms, vm_artifacts=vm_artifacts)
    future_summary = _verify_future_comparisons(first, arms=arms, artifacts=artifacts, vm_artifacts=vm_artifacts)

    bundles = [first]
    second = case_result.get("second_AB_world")
    if second is not None:
        second = _dict(second, "second AB bundle")
        second_parent_ref = _ref(second.get("parent_cut"), "second bundle parent")
        second_parent = _load_cut(second_parent_ref, artifacts, manifest, contract=contract)
        second_answer = _span(case_result["recurrent_policies"]["AB"]["preceding_answer_span"], "second answer", limit=len(second_parent["token_ids"]))
        _check_sites(second["sites"], program, second_parent, answer_span=second_answer, copula_token_id=copula)
        second_arms = _dict(second.get("arms"), "second arms")
        if set(second_arms) != required_arms:
            raise ValueError("second arm set changed")
        for name in sorted(second_arms):
            _verify_arm(second_arms[name], label=name, parent_ref=second_parent_ref, parent=second_parent,
                        artifacts=artifacts, vm_artifacts=vm_artifacts, manifest=manifest,
                        contract=contract, cache=cache)
        _verify_acts(second_arms, sites=second["sites"], parent_ref=second_parent_ref,
                     artifacts=artifacts, vm_artifacts=vm_artifacts)
        _verify_controls(second_arms)
        _verify_prediction(second, arms=second_arms, artifacts=artifacts, vm_artifacts=vm_artifacts, cache=cache)
        _verify_local_errors(second, arms=second_arms,
                             predictions=_dict(second["prediction_record"]["predictions"], "second prediction records"),
                             vm_artifacts=vm_artifacts)
        _verify_route_and_interaction(second, arms=second_arms, vm_artifacts=vm_artifacts)
        _verify_future_comparisons(second, arms=second_arms, artifacts=artifacts, vm_artifacts=vm_artifacts)
        bundles.append(second)

    event_summary = _verify_prediction_events(case_result, bundles)
    recurrent_summary = _verify_recurrent(case_result, program=program, artifacts=artifacts,
                                           vm_artifacts=vm_artifacts, manifest=manifest,
                                           contract=contract, cache=cache)
    semantic_summary = _verify_semantic_metrics(case_result, artifacts=artifacts, vm_artifacts=vm_artifacts)
    if case_result.get("semantic_cot_qualified") is not False or case_result.get("economical_query_formation_qualified") is not False:
        raise ValueError("read-composition report made an unsupported semantic-CoT claim")
    return {
        "schema": VERIFY_SCHEMA,
        "case_id": case_result.get("case", {}).get("id"),
        "mechanics_status": "valid",
        "instrument_status": "valid",
        "terminal_status": "not-assessed",
        "bench": bench_summary,
        "manifest_ref": manifest_ref,
        "first": {
            "parent_cut": parent_ref,
            "sites": program_sites,
            "prediction": prediction_summary,
            "local_errors": error_summary,
            "route": route_summary,
            "future_comparisons": future_summary,
        },
        "recurrent": recurrent_summary,
        "prediction_events": event_summary,
        "semantic": semantic_summary,
    }


def _case_root(results_root: Path, case_id: str) -> Path:
    direct = results_root / case_id
    if (direct / "selected.json").is_file():
        return direct
    matches = [path.parent for path in results_root.rglob("selected.json") if path.parent.name == case_id]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"cannot locate unique session tree for case {case_id!r}")


def verify_report(report: Mapping[str, Any], results_root: str | Path) -> dict[str, Any]:
    """Verify a decoded report and every case session under ``results_root``."""

    _finite(report)
    report = _dict(report, "report")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("read-composition report schema mismatch")
    if report.get("semantic_cot_qualified") is not False:
        raise ValueError("read-composition report cannot qualify semantic CoT")
    try:
        program = FrozenReadProgram.from_dict(report["program"])
    except Exception as error:
        raise ValueError("report FrozenReadProgram is invalid") from error
    results_root = Path(results_root)
    case_values = _list(report.get("cases"), "report cases")
    if not case_values:
        raise ValueError("report contains no cases")
    seen = set()
    cases = []
    for value in case_values:
        case = _dict(value, "report case")
        case_id = _dict(case.get("case"), "report case declaration").get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("report case identity is malformed or duplicated")
        seen.add(case_id)
        cases.append(verify_case(case, _case_root(results_root, case_id), program=program))
    return {
        "schema": VERIFY_SCHEMA,
        "report_schema": REPORT_SCHEMA,
        "mechanics_status": "valid",
        "instrument_status": "valid",
        "terminal_status": "not-assessed",
        "semantic_cot_qualified": False,
        "program_fingerprint": program.fingerprint,
        "cases": cases,
    }


def verify(root: str | Path) -> dict[str, Any]:
    """Verify ``report.json`` plus all session trees beneath one result root."""

    root = Path(root)
    report_path = root / "report.json" if root.is_dir() else root
    report = json.loads(report_path.read_text())
    return verify_report(report, report_path.parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="recovered result directory or report.json")
    args = parser.parse_args(argv)
    result = verify(args.root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
