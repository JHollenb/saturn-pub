import runpy
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock


def helper(name):
    return runpy.run_path(str(Path(__file__).parent.parent / "qualification" / name))


def test_native_attention_reconstruction_and_source_selection():
    torch.manual_seed(31)
    block = Flux2SingleTransformerBlock(8, 2, 4, mlp_ratio=2).eval()
    modulation = torch.randn(1, 24)
    angles = torch.randn(5, 2).repeat_interleave(2, dim=-1)
    rope = angles.cos(), angles.sin()
    rotate = helper("flux_functional_denoiser.py")["_rotate"]
    reconstruct = helper("flux_attention_sources.py")["attention_sources"]
    histories = {}
    for name in ("base", "changed"):
        ports = {}
        def q(module, args, result):
            ports["q"] = result.detach().clone()
        def k(module, args, result):
            ports["k"] = result.detach().clone()
        def v(module, args, result):
            ports["v"] = result[..., :24].chunk(3, dim=-1)[2].unflatten(-1, (2, 4)).detach().clone()
        def out(module, args):
            ports["native"] = args[0][..., :8].detach().clone()
        hooks = [block.attn.norm_q.register_forward_hook(q), block.attn.norm_k.register_forward_hook(k),
                 block.attn.to_qkv_mlp_proj.register_forward_hook(v), block.attn.to_out.register_forward_pre_hook(out)]
        try:
            with torch.inference_mode():
                block(torch.randn(1, 5, 8), None, modulation, image_rotary_emb=rope)
        finally:
            for hook in hooks:
                hook.remove()
        ports["q"], ports["k"] = rotate(ports["q"], rope), rotate(ports["k"], rope)
        histories[name] = ports
    for mask, name in (("000", "changed"), ("111", "base")):
        got = reconstruct(**histories, text_rows=2, mask=mask)
        assert torch.equal(got["attention"], histories[name]["native"])
    mixed = reconstruct(**histories, text_rows=2, mask="010")
    assert torch.equal(mixed["q"], histories["changed"]["q"])
    assert torch.equal(mixed["k"][:, :2], histories["base"]["k"][:, :2])
    assert torch.equal(mixed["v"][:, 2:], histories["changed"]["v"][:, 2:])
    with pytest.raises(ValueError, match="mask"):
        reconstruct(**histories, text_rows=2, mask="bad")


def test_frozen_contexts_and_predictions_are_explicit():
    historical = {"H-ordering": {"merge.010": "base"}}
    plan = helper("submit_flux_route_replication.py")["campaign"]({"model": "klein", "model_path": "/model", "expected_checkpoint_sha256": "a" * 64, "hypotheses": historical})
    assert [(row["id"], row["seed"]) for row in plan["cases"]] == [("original37", 37), ("paraphrase31", 31), ("paraphrase37", 37)]
    assert sum(row["attention_sources"] for row in plan["cases"]) == 1
    assert plan["historical_record"]["sealed_hypotheses"] == historical
    assert plan["replication_expectations"]["H-ordering"]["merge.010"] == "changed"
    assert plan["replication_expectations"]["H-ordering"]["merge.110"] == "base"


def test_replication_submission_keeps_offline_cache_and_one_finite_lease(tmp_path, monkeypatch):
    import json

    import mdb.job_support
    import mdb.payload

    module = helper("submit_flux_route_replication.py")
    plan = module["campaign"]({"model": "klein", "model_path": "/model", "expected_checkpoint_sha256": "a" * 64,
        "hypotheses": {"H-ordering": {"merge.010": "base"}}})
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(plan))
    staged = {}
    monkeypatch.setattr(mdb.payload, "stage_worker", lambda root, **kwargs: staged.update(kwargs))
    monkeypatch.setattr(mdb.payload, "worker_command", lambda name: ["uv", "run", "--offline", "python", name])

    def launch(command, **kwargs):
        config = kwargs["request"]["config"]
        assert "--offline" in command
        assert kwargs["request"] == staged["request"]
        assert config["max_images"] == (17 if config["attention_sources"] else 11)
        assert config["campaign_sha256"] == module["sha"](path)
        assert config["hypotheses"] == plan["replication_expectations"]
        assert kwargs["preflight"] and not kwargs["retry_on_kill"]
        assert kwargs["needs"] == {"cuda": True}
        assert kwargs["request"]["scheduling_contract"]["model_loads"] == 1
        assert kwargs["request"]["scheduling_contract"]["logical_branches"] == config["max_images"]
        return "job-test"

    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", launch)
    for case in plan["cases"]:
        assert module["stage"](tmp_path / case["id"], path, case["id"])["job_id"] == "job-test"
