"""Public Saturn training transactions and MDB native carrier assays.

This experiment orchestrates two independent public packages. MDB owns only
declared native consumer inputs; Saturn owns the complete mutable training
closure. Custom models have no inherited pretrained-family qualification.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from mdb.research_session import NativeResearchSession

from saturn_pub import Act, Adapter, NumericalContract, Session
from saturn_pub.training import (
    EvaluationScores,
    ImmutableDirectoryReceiptStore,
    PromotionObjective,
    PromotionPolicy,
    TrainingTransaction,
)
from saturn_pub.values import clone, describe, digest


def _name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("artifact name must contain only letters, digits, dot, dash, underscore")
    return value


def _json(path: Path, value: Any) -> None:
    """Publish one immutable raw record; duplicate names are refused."""
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)


def _weights(model: torch.nn.Module) -> str:
    return digest(describe(dict(model.state_dict())))


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def language_observation(
    logits: torch.Tensor, targets: torch.Tensor, generated: torch.Tensor | None = None
) -> dict[str, Any]:
    """Summarize teacher-forced suffix loss and model-predicted byte history."""
    if logits.shape[:-1] != targets.shape or targets.dtype != torch.long:
        raise ValueError("logits/targets do not form a typed language-model suffix")
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    )
    observation = {
        "bpb": float(loss.mean() / math.log(2)),
        "per_token_bits": (loss / math.log(2)).detach().cpu().tolist(),
        "logits_sha256": digest(describe(logits)),
        "target_tokens": targets.numel(),
    }
    if generated is not None:
        observation["generated_history"] = generated.detach().cpu().tolist()
    return observation


class NativeInputAdapter(Adapter):
    """A Saturn input-cut adapter around the caller's fixed native consumer.

    Only input tensors are addressable. The consumer runs exactly once per
    transition; output tensors and observation become retained evidence. This
    does not claim to pause inside arbitrary model operators.
    """

    def __init__(self, *, model, identity, initial_state, writable, consumer):
        self.model, self.consumer = model, consumer
        self.model_identity = identity["weights_sha256"]
        self.execution = {**identity, "adapter": "spf-native-input-cut-v1"}
        self.input_names = tuple(sorted(initial_state))
        self.writable = tuple(writable)
        self.spec = {key: (value.shape, value.dtype) for key, value in initial_state.items()}

    def boundary(self, state):
        return "native-output" if "output" in state else "native-input"

    def addresses(self, state):
        return self.writable

    def validate(self, state):
        expected = set(self.input_names)
        if "output" in state:
            expected |= {"output", "observation"}
        if set(state) != expected:
            raise ValueError("native input-cut closure changed")
        for key, (shape, dtype) in self.spec.items():
            value = state[key]
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"native tensor shape/dtype mismatch: {key}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"non-finite native tensor: {key}")

    def validate_execution(self):
        if _weights(self.model) != self.model_identity:
            raise ValueError("native consumer model changed after input capture")

    def numerical_contract(self):
        return NumericalContract(program="spf-native-input-cut-v1")

    def advance(self, state):
        if "output" in state:
            raise ValueError("bounded native consumer already completed")
        output, observation = self.consumer({key: clone(state[key]) for key in self.input_names})
        return {**state, "output": clone(output), "observation": clone(observation)}


class PublicRuntime:
    """One resident model, public transactional training, immutable evidence."""

    def __init__(
        self,
        model,
        optimizer,
        cursor,
        root,
        identity,
        *,
        generators=None,
        mutable=None,
        max_interval_steps=4,
        policy=None,
    ):
        self.model, self.optimizer = model, optimizer
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = self.root / "training_snapshots"
        self.snapshots.mkdir(exist_ok=True)
        self.identity = (
            dict(identity) if isinstance(identity, Mapping) else {"model_id": str(identity)}
        )
        self.identity.setdefault("model_id", type(model).__name__)
        self.identity.setdefault("family", "custom-byte-language-model")
        self.identity.setdefault("numerical_contract", "custom-native-same-program-exact-v1")
        if isinstance(cursor, dict):

            def capture_cursor():
                return copy.deepcopy(cursor)

            def restore_cursor(state):
                cursor.clear()
                cursor.update(copy.deepcopy(state))
        elif callable(getattr(cursor, "state_dict", None)) and callable(
            getattr(cursor, "load_state_dict", None)
        ):
            capture_cursor, restore_cursor = cursor.state_dict, cursor.load_state_dict
        else:
            raise ValueError("cursor must be a dict or expose state_dict/load_state_dict")
        self.controller = TrainingTransaction(
            controller_id="spf-language-model",
            model=model,
            optimizer=optimizer,
            cursor_capture=capture_cursor,
            cursor_restore=restore_cursor,
            mutable=mutable,
            torch_generators=generators,
            max_interval_steps=max_interval_steps,
            policy=policy
            or PromotionPolicy(
                "lexicographic", (PromotionObjective("validation", "bpb", "minimize"),)
            ),
            receipt_store=ImmutableDirectoryReceiptStore(self.root / "training_receipts"),
        )
        self.last_candidate_archive = None

    def capture(self):
        return self.controller.capture()

    def restore(self, snapshot):
        return self.controller.restore(snapshot)

    def archive(self, snapshot, label):
        """Retain executable accepted/rejected closure, beyond score receipts."""
        label = _name(label)
        snapshot.verify()
        path = self.snapshots / f"{label}-{snapshot.fingerprint}.pt"
        if not path.exists():
            with path.open("xb") as stream:
                torch.save(snapshot.to_state_dict(), stream)
        return {
            "path": str(path),
            "fingerprint": snapshot.fingerprint,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def load_snapshot(self, path):
        # The supported snapshot serialization constructor is available on the
        # public boundary's returned snapshot type; no private Saturn imports.
        snapshot_type = type(self.capture())
        return snapshot_type.from_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)
        )

    def establish_baseline(self, evaluator):
        archive = self.archive(self.capture(), "baseline")
        receipt = self.controller.establish_baseline(evaluator)
        _json(self.root / "baseline-archive.json", archive)
        return receipt

    baseline = establish_baseline

    def run_interval(self, *, interval_id, steps, train_step, evaluator):
        interval_id = _name(interval_id)
        self.last_candidate_archive = None

        def bounded_step(index):
            try:
                return train_step(index)
            except Exception:
                # Keep a partially executed future when its closure remains
                # serializable. The public controller still restores/reports
                # the original error if the partial snapshot is invalid.
                try:
                    self.last_candidate_archive = self.archive(
                        self.capture(), interval_id + "-error"
                    )
                    _json(
                        self.root / f"{interval_id}-error-archive.json", self.last_candidate_archive
                    )
                except Exception:
                    pass
                raise

        def inspect_candidate():
            self.last_candidate_archive = self.archive(self.capture(), interval_id)
            scores = EvaluationScores.from_value(evaluator()).to_dict()
            scores["evidence"] = {
                **scores.get("evidence", {}),
                "candidate_archive": self.last_candidate_archive,
                "terminal_status": "not-assessed",
            }
            return scores

        return self.controller.run_interval(
            interval_id=interval_id,
            steps=steps,
            train_step=bounded_step,
            evaluator=inspect_candidate,
        )

    evaluate_candidate = run_interval

    def native_panel(
        self,
        *,
        initial_state,
        writable,
        consumer,
        panel_id="final",
        doses=(0.0, 0.5, 1.0),
        resident_bytes=64 << 20,
    ):
        """Run paired native suffixes from one durable MDB input parent.

        ``consumer(state) -> (tensor_output_tree, JSON_observation)`` must use
        only declared inputs and the current fixed model. It should return full
        suffix logits and model-predicted generated histories. All declared
        writable carrier tensors are scaled by ``1 - dose``. Scores never gate
        exploratory rows; exact no-op/parent-restoration are mechanics checks.
        """
        panel_id = _name(panel_id)
        writable = tuple(writable)
        if not writable or not set(writable).issubset(initial_state):
            raise ValueError("declare at least one writable carrier input")
        if not initial_state or any(
            not isinstance(value, torch.Tensor) for value in initial_state.values()
        ):
            raise ValueError("native input cut must be a flat named tensor closure")
        doses = tuple(float(dose) for dose in doses)
        if any(not math.isfinite(dose) for dose in doses):
            raise ValueError("carrier doses must be finite")
        if any(not initial_state[key].is_floating_point() for key in writable):
            raise ValueError("carrier scale intervention requires floating-point inputs")
        closure = self.capture()
        fixed_weights = _weights(self.model)
        identity = {
            **self.identity,
            "weights_sha256": fixed_weights,
            "claim_boundary": "custom-model-native-input-cuts; family qualification separate",
        }
        panel_root = self.root / "native_panels" / panel_id
        panel_root.parent.mkdir(exist_ok=True)
        panel_root.mkdir(exist_ok=False)
        shape_types = {key: (value.shape, value.dtype) for key, value in initial_state.items()}

        def validate(state):
            for key, (shape, dtype) in shape_types.items():
                if state[key].shape != shape or state[key].dtype != dtype:
                    raise ValueError(f"native input shape/dtype mismatch: {key}")

        def native(state):
            with torch.no_grad():
                output, observation = consumer(state)
            if not isinstance(observation, dict):
                raise TypeError("native observation must be JSON dictionary")
            json.dumps(observation, allow_nan=False)
            return clone(output), observation

        rows, outputs, session = [], {}, None
        physical_calls = 0
        self.model.eval()
        try:
            session = NativeResearchSession(
                root=panel_root / "mdb",
                identity=identity,
                initial_state=clone(initial_state),
                writable=writable,
                consumer=native,
                validator=validate,
                device=str(next(iter(initial_state.values())).device),
                resident_bytes=resident_bytes,
            )

            def run(name, *, updates=None, parent="parent", dose=None):
                nonlocal physical_calls
                physical_calls += 1
                output, record = session.run(
                    name,
                    parent=parent,
                    updates=updates,
                    metadata={
                        "dose": dose,
                        "stage": "discovery",
                        "terminal_status": "not-assessed",
                    },
                )
                outputs[name] = clone(output)
                artifact = panel_root / f"{name}-outputs.pt"
                with artifact.open("xb") as stream:
                    torch.save(clone(output), stream)
                row = {
                    "arm": name,
                    "dose": dose,
                    "parent": parent,
                    **record,
                    "output_artifact": str(artifact),
                    "output_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
                _json(panel_root / f"{name}-row.json", row)
                rows.append(row)
                return output

            native_output = run("native")
            no_op = run(
                "no-op", updates={key: clone(initial_state[key]) for key in writable}, dose=0.0
            )
            no_op_exact = _equal(native_output, no_op)
            ablated_name = None
            for index, dose in enumerate(doses):
                name = f"dose-{index}"
                if dose == 0:
                    row = {
                        "arm": name,
                        "dose": dose,
                        "alias_of": "no-op",
                        "physical_consumer_calls": 0,
                        "terminal_status": "not-assessed",
                    }
                    _json(panel_root / f"{name}-row.json", row)
                    rows.append(row)
                    outputs[name] = clone(no_op)
                    continue
                try:
                    run(
                        name,
                        updates={key: initial_state[key] * (1.0 - dose) for key in writable},
                        dose=dose,
                    )
                    if dose == 1:
                        ablated_name = name
                except Exception as error:
                    row = {
                        "arm": name,
                        "dose": dose,
                        "instrument_status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "terminal_status": "not-assessed",
                    }
                    _json(panel_root / f"{name}-error.json", row)
                    rows.append(row)
            if ablated_name is None:
                ablated_name = "full-ablation"
                run(
                    ablated_name,
                    updates={key: torch.zeros_like(initial_state[key]) for key in writable},
                    dose=1.0,
                )
            repair_output = run(
                "repair",
                parent=ablated_name,
                updates={key: clone(initial_state[key]) for key in writable},
            )
            session.restore("parent")
            restored_output = run("restored-parent")
            # Independently exercise a typed public Saturn input-cut session.
            adapter = NativeInputAdapter(
                model=self.model,
                identity=identity,
                initial_state=initial_state,
                writable=writable,
                consumer=native,
            )
            saturn = Session(adapter, clone(initial_state))
            cut = saturn.capture()
            branch = saturn.fork(cut)
            for key in writable:
                branch.apply(Act.replace(key, clone(initial_state[key]), name="no-op"))
            branch.continue_()
            physical_calls += 1
            saturn.restore(cut)
            typed_exact = _equal(branch.capture(retain=False).payload["output"], native_output)
            _json(
                panel_root / "saturn-input-cut.json",
                {
                    "parent": cut.manifest(),
                    "parent_fingerprint": cut.fingerprint,
                    "branch_receipts": [receipt.to_dict() for receipt in branch.receipts],
                    "restore_receipts": [receipt.to_dict() for receipt in saturn.receipts],
                },
            )
            if _weights(self.model) != fixed_weights:
                raise ValueError("native panel mutated frozen model state")
            mechanics = {
                "no_op_exact": no_op_exact,
                "repair_exact": _equal(native_output, repair_output),
                "restore_exact": _equal(native_output, restored_output),
                "saturn_no_op_exact": typed_exact,
            }
            summary = {
                "schema": "spf-public-native-panel-v1",
                "identity": identity,
                "rows": rows,
                "mechanics": mechanics,
                "mechanics_status": "valid" if all(mechanics.values()) else "invalid",
                "instrument_status": "partial"
                if any(row.get("instrument_status") == "error" for row in rows)
                else "ran",
                "terminal_status": "not-assessed",
                "physical_consumer_calls": physical_calls,
                "session_root": str(session.root),
                "memory": session.inspect_memory(),
                "limitations": [
                    "declared input-cut replay; no arbitrary intra-operator pause",
                    "custom model mechanics do not extend qualified pretrained numerical claims",
                ],
            }
            _json(panel_root / "summary.json", summary)
            if summary["mechanics_status"] != "valid":
                raise RuntimeError("native input-cut no-op/repair/restore mechanics mismatch")
            return summary
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                self.restore(closure)


def initialize_runtime(model, optimizer, cursor, root, identity, **kwargs):
    return PublicRuntime(model, optimizer, cursor, root, identity, **kwargs)
