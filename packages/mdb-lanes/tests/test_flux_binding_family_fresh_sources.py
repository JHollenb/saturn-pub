"""Model-free qualification of fresh input custody and original source policy."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from saturn_pub import Act, Session
from saturn_pub.contracts import ExecutionPoint
from saturn_pub.core import Adapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb_lanes.forecast_transaction import ForecastTransaction

ROOT = Path(__file__).parents[1]
FILE = ROOT / "qualification/flux_binding_family_fresh_sources.py"
W = runpy.run_path(str(FILE))
STAGE = runpy.run_path(str(FILE.with_name("stage_flux_binding_family_fresh_sources.py")))
BASE = runpy.run_path(str(FILE.with_name("flux_centered_write_campaign.py")))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def bits(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def tokens(context):
    """Synthetic whitespace/punctuation tokenizer; no pretrained vocabulary."""
    result = {}
    for role, prompt in context["prompts"].items():
        ids, mask, offsets, pieces = [0] * 512, [0] * 512, [[0, 0] for _ in range(512)], ["<|pad|>"] * 512
        for row, match in enumerate(re.finditer(r"\w+|[^\w\s]", prompt), start=2):
            piece = match.group()
            ids[row] = int(hashlib.sha256(piece.encode()).hexdigest()[:8], 16)
            mask[row], offsets[row], pieces[row] = 1, [10 + match.start(), 10 + match.end()], piece
        spans = []
        for attribute in context["attributes"]:
            start = 10 + prompt.index(attribute)
            spans.append(dict(attribute=attribute, start=start, end=start + len(attribute),
                rows=[row for row, (a, b) in enumerate(offsets) if a < start + len(attribute) and b > start]))
        result[role] = dict(ids=ids, mask=mask, offsets=offsets, pieces=pieces, attribute_spans=spans)
    return result


def config(rule_sha):
    return dict(schema=W["SCHEMA"], output_name=W["RESULT"], contexts=W["contexts"](), source_policy=W["POLICY"],
        expected_model_identity=W["MODEL_ID"], expected_counts=W["contract"](), no_joint0_outputs=True,
        retries=0, nested_mrun=False, height=512, width=512, steps=4, batch_size=1, dtype="bfloat16", device="cuda",
        family_rule=dict(schema="fixture-frozen-rule", fresh_contexts=W["contexts"]()), family_rule_sha256=rule_sha)


class InputOnly(Adapter):
    """Public StateCut fixture; CUDA is recorded provenance, never executed here."""
    model_identity = W["MODEL_ID"]

    def __init__(self, device="cpu"):
        self.execution = dict(family="flux2-klein", dtype="torch.bfloat16", device=dict(type=device, index=None),
                              profile="explicit model-free CAS fixture; not CUDA qualification")
        self.validations = 0

    def validate_execution(self):
        self.validations += 1

    def validate(self, state):
        assert state["text"].shape == (1, 512, 4) and state["text"].dtype == torch.bfloat16

    def addresses(self, state):
        return ("image", "text", "conditioning")

    def boundary(self, state):
        return f"step:{state['step']}/{state['phase']}:{state['cursor']}"

    def point(self, state):
        return ExecutionPoint(family="flux2-klein", logical_step=state["step"], phase=state["phase"],
            operator=state["phase"], index=state["cursor"] if state["phase"] == "joint" else None,
            edge="before", local_clock=state["cursor"], next_operation="joint_block.0")

    def advance(self, state):
        raise AssertionError("Input source loading must execute zero model transitions")


def inventory(root, publication):
    records = {path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=W["sha"](path))
               for path in root.rglob("*") if path.is_file() and path.name != "files.json"}
    write(root / "files.json", dict(job_id=publication["job_id"], request_sha256=publication["request_sha256"], files=records))
    publication["files_manifest_sha256"] = W["sha"](root / "files.json")


def source_fixture(tmp_path):
    root, output = tmp_path / "source", tmp_path / "consumer"
    root.mkdir()
    output.mkdir()
    namespace, adapter = W["helpers"](), InputOnly("cuda")
    rule = dict(schema="fixture-frozen-rule", fresh_contexts=W["contexts"]())
    write(root / "family-rule.json", rule)
    rule_sha = W["sha"](root / "family-rule.json")
    cfg = config(rule_sha)
    request = dict(config=cfg, source_hashes={FILE.name: W["sha"](FILE)})
    request["request_sha256"] = W["request_identity"](request)
    write(root / "request.json", request)
    publication = dict(output_root=str(root), job_id="job-fresh-input-fixture", request_sha256=request["request_sha256"])
    source_store, roots, selected, token_rows = LocalStore(root / "state-store"), {}, {}, {}
    for index, context in enumerate(W["contexts"]()):
        cid = context["id"]
        token_rows[cid] = tokens(context)
        selected[cid] = W["select_addresses"](context, token_rows[cid], namespace)
        text = (torch.arange(512 * 4).reshape(1, 512, 4) % 17).to(torch.bfloat16)
        text[:, 0, 0] = -0.
        common = dict(text=text, image=torch.full((1, 2, 4), index, dtype=torch.bfloat16),
            conditioning=torch.ones(1, 512, 6, dtype=torch.bfloat16), context=dict(ids=torch.arange(4)),
            step=2, phase="joint", cursor=0)
        master = Session(adapter, common)
        parent = master.capture(retain=False)
        inputs = {}
        for role, rows in (("P", selected[cid]["A"]), ("Q", selected[cid]["B"])):
            branch = master.fork(parent)
            payload = text.clone()
            payload[:, rows] += .5
            branch.apply(Act.replace("text", payload, name="source-only-row-SET"))
            cut = branch.capture(retain=False)
            source_store.save(cut)
            inputs[role] = dict(BASE["cut_record"](cut), execution=dict(cut.execution))
            branch.restore(parent)
            assert branch.capture(retain=False).fingerprint == parent.fingerprint
        roots[cid] = dict(inputs=inputs, support=selected[cid]["S"], wrong_rows=selected[cid]["wrong_rows"], known={},
            base_conditioning=namespace["flux_raw_to_image.py"]["_save_tensor"](root, f"inputs/{cid}/base-conditioning.npy", common["conditioning"]))
    tx = ForecastTransaction(root / "source-transaction", model_identity=adapter.model_identity,
        input_identity="synthetic token-only source fixture", source_identity=rule_sha,
        execution_scope="Model-free fixture retains inputs only", frontier=dict(kind="raw-input"))
    tx.frontier("raw-contexts-and-frozen-rule", dict(contexts=W["contexts"](), family_rule=rule))
    tx.predict("prefix-policy", lambda _ctx: dict(predictions=dict(counts=W["contract"]())))
    seal = tx.seal()
    tx.execute("native-prefix-inputs", lambda _ctx: dict(observations="synthetic CAS input fixture, no model calls"))
    tx.close()
    write(root / "summary.json", dict(schema=W["SCHEMA"], job_id=publication["job_id"], request_sha256=request["request_sha256"],
        source_policy=W["POLICY"], family_rule_sha256=rule_sha, counters=W["contract"](), seal_sha256=seal["seal_sha256"]))
    write(root / "fresh-inputs.json", dict(schema=W["SCHEMA"], contexts=W["contexts"](), roots=roots,
        token_addresses=dict(tokens=token_rows, addresses=selected), family_rule_sha256=rule_sha,
        source_policy=W["POLICY"], counters=W["contract"]()))
    (root / "unconsumed-byte-fixture").write_bytes(b"Never copied by input-only loader")
    inventory(root, publication)
    consumer = dict(fresh_source=publication, family_rule=rule, family_rule_file_sha256=rule_sha)
    return consumer, output, LocalStore(output / "state-store"), InputOnly(), BASE


def test_standalone_definitions_and_installed_entrypoint_are_framework_free():
    code = "import runpy,sys; runpy.run_path(sys.argv[1]); assert not {'torch','numpy','saturn_pub','mdb'} & sys.modules.keys()"
    for path in (FILE, FILE.with_name("stage_flux_binding_family_fresh_sources.py")):
        subprocess.run([sys.executable, "-I", "-B", "-c", code, str(path)], check=True)
        subprocess.run([sys.executable, "-I", "-B", str(path), "--help"], check=True, capture_output=True)
    assert set(W["helpers"]()) == set(W["HELPERS"])


def test_rule_bytes_and_complete_fixed_cohort_are_required_before_source_work(tmp_path):
    path = tmp_path / "family-rule.json"
    write(path, dict(schema="fixture-frozen-rule", fresh_contexts=W["contexts"]()))
    cfg = dict(config(W["sha"](path)), family_rule_path=str(path))
    assert W["validate_rule"](cfg) == W["sha"](path)
    assert W["validate_config"](cfg, files=False) == W["contract"]()
    changed = copy.deepcopy(cfg)
    changed["family_rule"]["fresh_contexts"][0]["seed"] += 1
    with pytest.raises(ValueError, match="exact six fresh"):
        W["validate_rule"](changed, files=False)
    path.write_text(json.dumps(cfg["family_rule"], indent=2))
    with pytest.raises(ValueError, match="bytes differ"):
        W["validate_rule"](cfg)
    with pytest.raises(ValueError, match="digest required"):
        W["validate_rule"](dict(cfg, family_rule_sha256="0" * 64), files=False)


def test_token_only_addresses_never_use_padding_or_special_rows_and_transport_retains_signed_zeros():
    namespace = W["helpers"]()
    context = W["contexts"]()[0]
    record = tokens(context)
    selected = W["select_addresses"](context, record, namespace)
    assert len(selected["S"]) == len(selected["wrong_rows"]) == 4
    assert not set(selected["S"]) & set(selected["wrong_rows"])
    assert selected["B"] == [row for row in selected["B_metadata"]["candidates"]][:2]
    assert all(record["base"]["mask"][row] and record["source"]["mask"][row] for row in selected["wrong_rows"])
    base = torch.zeros(1, 512, 4, dtype=torch.bfloat16)
    base[:, 0] = -0.
    amid = base.clone()
    amid[:, selected["A"]] = torch.tensor([.5, -1., 2., -.25], dtype=torch.bfloat16)
    pair, recipes = W["pair_text"](base, amid, selected, namespace["flux_binding_route_payload_math.py"])
    assert bits(pair["P"], amid)
    assert bits(pair["Q"][:, selected["B"]], amid[:, selected["A"]])
    complement = [row for row in range(512) if row not in selected["S"]]
    assert bits(pair["P"][:, complement], pair["Q"][:, complement]) and torch.signbit(pair["Q"][:, 0]).all()
    assert recipes["P"]["target_l2"] == recipes["Q"]["landed_l2"]
    zero, _ = W["pair_text"](base, base, selected, namespace["flux_binding_route_payload_math.py"])
    assert bits(zero["P"], base) and bits(zero["Q"], base)
    amid[:, 0, 0] = 1
    with pytest.raises(ValueError, match="outside A"):
        W["pair_text"](base, amid, selected, namespace["flux_binding_route_payload_math.py"])


def test_shared_prefix_stop_guard_blocks_joint0_entry_before_any_toy_module_call():
    calls = []
    module = torch.nn.Identity()
    point = dict(logical_step=1, phase="joint", index=4, edge="before")
    handle = module.register_forward_pre_hook(lambda *_args: W["prefix_point"](point, phase="joint", index=4) and None)
    observed = module.register_forward_hook(lambda *_args: calls.append("native toy module"))
    module(torch.ones(1))
    point.update(logical_step=2, phase="joint", index=0)
    with pytest.raises(ValueError, match="stop frontier"):
        module(torch.ones(1))
    assert calls == ["native toy module"]
    handle.remove()
    observed.remove()
    assert W["prefix_point"](dict(logical_step=2, phase="project", index=None, edge="before"))
    with pytest.raises(ValueError, match="full-clock"):
        W["prefix_point"](dict(logical_step=1, phase="single", index=4, edge="before"), phase="joint", index=4)


def test_actual_public_input_publication_loads_six_pairs_without_forward_or_source_mutation(tmp_path, monkeypatch):
    cfg, output, store, adapter, base = source_fixture(tmp_path)
    source_root = Path(cfg["fresh_source"]["output_root"])
    before = {path: W["sha"](path) for path in source_root.rglob("*") if path.is_file()}
    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_a, **_kw: pytest.fail("source loader called model"))
    roots, provenance = W["load_sources"](cfg, output, store, adapter, base)
    assert set(roots) == {row["id"] for row in W["contexts"]()} and provenance["model_calls"] == 0
    assert provenance["transaction"]["seal_sha256"] and adapter.validations > 0
    assert len(list((store.root / "cuts").glob("*.json"))) == 12
    for row in roots.values():
        p, q = (store.load(row["inputs"][role]["cut"], device="cpu") for role in ("P", "Q"))
        assert p.execution["device"]["type"] == "cuda" and p.payload["text"].device.type == "cpu"
        assert describe({key: value for key, value in p.payload.items() if key != "text"}) == describe({key: value for key, value in q.payload.items() if key != "text"})
        assert bits(base["load_tensor"](output, row["base_conditioning"], "cpu"), q.payload["conditioning"])
        assert row["known"] == {}
    assert "unconsumed-byte-fixture" not in provenance["consumed_members"]
    assert not (output / "unconsumed-byte-fixture").exists()
    assert all(W["sha"](path) == digest for path, digest in before.items())


@pytest.mark.parametrize("change", ("rule", "blob", "clock", "profile", "model", "outside-support"))
def test_hostile_saved_inputs_rule_profile_clock_and_bytes_reject(tmp_path, change):
    cfg, output, store, adapter, base = source_fixture(tmp_path)
    root = Path(cfg["fresh_source"]["output_root"])
    if change == "rule":
        cfg["family_rule_file_sha256"] = "b" * 64
    elif change == "blob":
        path = next((root / "state-store/blobs").glob("*.safetensors"))
        path.write_bytes(path.read_bytes() + b"corruption")
    elif change == "model":
        adapter.model_identity = "foreign-original-checkpoint"
    elif change == "profile":
        adapter.execution["profile"] = "different kernel/profile"
    else:
        table = W["read"](root / "fresh-inputs.json")
        cid = W["contexts"]()[0]["id"]
        record = table["roots"][cid]["inputs"]["P"]
        original = LocalStore(root / "state-store").load(record["cut"], device="cpu")
        state = copy.deepcopy(dict(original.payload))
        if change == "clock":
            state["cursor"] = 1
        else:
            state["text"][:, 0, 0] = 1.
        source_adapter = InputOnly("cuda")
        changed = Session(source_adapter, state, parent=original.parent).capture(retain=False)
        LocalStore(root / "state-store").save(changed)
        table["roots"][cid]["inputs"]["P"] = dict(BASE["cut_record"](changed), execution=dict(changed.execution))
        write(root / "fresh-inputs.json", table)
        inventory(root, cfg["fresh_source"])
    with pytest.raises(ValueError, match="rule bytes|member changed|frontier differs|execution profile|model identity|outside support"):
        W["load_sources"](cfg, output, store, adapter, base)


def test_collision_check_reads_workload_metadata_and_ignores_rule_proposals(tmp_path):
    context = W["contexts"]()[0]
    write(tmp_path / "rule/request.json", dict(config=dict(contexts=["historical"], family_rule=dict(fresh_contexts=[context]))))
    write(tmp_path / "qualification-01/request.json", dict(config=dict(contexts=[context])))
    write(tmp_path / "historical/plan.json", dict(contexts=[dict(context, seed=1)]))
    (tmp_path / "audit.json").write_bytes(b"outcomes forbidden to collision check")
    assert STAGE["collision_check"](tmp_path, W["contexts"]())["collisions"] == []
    write(tmp_path / "prior/request.json", dict(config=dict(contexts=[context])))
    receipt = STAGE["collision_check"](tmp_path, W["contexts"]())
    assert [row["context"] for row in receipt["collisions"]] == [context["id"]]
    assert "audit.json" not in receipt["checked_metadata"]


def test_staged_command_uses_qualified_public_wheels_and_explicit_entrypoint():
    request = dict(runtime_wheels={"mdb.whl": "a" * 64, "saturn.whl": "b" * 64, "execution.whl": "c" * 64},
        execution_environment=dict(python="3.14.6", uv_link_mode="copy", uv_cache_dir="/cache", torch_home="/torch",
            extra_model_roots="/models", requirements=["numpy==2.5.4", "psutil==7.2.2", "pyyaml==6.0.3"]))
    command = STAGE["command"](request)
    assert "--no-project" in command and "--offline" in command
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in command
    assert command[-5:] == ["-m", "mdb_lanes.worker_bootstrap", FILE.name, "--config", "saturn_run_config.json"]
    assert command.count("--with") == 6 and not any("common" in item for item in command)
