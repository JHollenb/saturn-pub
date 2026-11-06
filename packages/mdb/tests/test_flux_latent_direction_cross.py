"""Small mechanics checks; these do not count as real-model evidence."""
import runpy
from pathlib import Path

import numpy as np
import pytest

Q = Path(__file__).resolve().parents[1] / "qualification"
W = runpy.run_path(str(Q / "flux_latent_direction_cross.py"))


def test_rounding_matches_installed_torch_bf16():
    import torch
    x = np.random.default_rng(2).normal(size=4096).astype(np.float32)
    expected = torch.from_numpy(x).bfloat16().float().numpy()
    np.testing.assert_array_equal(W["bf16"](x), expected)


def test_unit_direction_recovers_source_and_dose_is_not_retuned():
    native = W["bf16"](np.array([.2, .4, .8, 1.], dtype=np.float32))
    predicted = W["bf16"](native + np.array([.01, -.01, .02, -.03], dtype=np.float32))
    delta = predicted.astype(np.float64) - native.astype(np.float64)
    value, detail = W["compose"](native, delta, W["rms"](delta))
    np.testing.assert_array_equal(value, predicted)
    assert detail["scale"] == 1
    value, detail = W["compose"](native, delta, W["rms"](delta) * .6)
    assert detail["scale"] == .6
    assert detail["effective_rms"] == W["rms"](value.astype(np.float64) - native)
    assert detail["effective_rms"] != detail["requested_rms"]
    assert 0 < detail["effective_source_cosine"] <= 1


@pytest.mark.parametrize("dose", [0., -1., float("nan")])
def test_invalid_nominal_dose_rejected(dose):
    with pytest.raises(ValueError):
        W["compose"](np.ones(3, np.float32), np.ones(3), dose)


def test_zero_direction_rejected():
    with pytest.raises(ValueError, match="zero direction"):
        W["compose"](np.ones(3, np.float32), np.zeros(3), .01)


def kind_at(local):
    if local == 0:
        return "project", None
    if local < 6:
        return "joint", local - 1
    if local < 26:
        return "single", local - 6
    return ("readout", None) if local == 26 else ("commit", None)


def test_complete_own_state_suffix_and_retention(tmp_path):
    calls = []
    def events(latent, conditioning, img_ids, txt_ids, schedule, binding, load):
        calls.append(latent.copy())
        image, text = latent.copy(), conditioning.copy()
        yield "project", None, {"image": image, "text": text, "context": {"temb": np.ones(1, np.float32)}}
        for i in range(5):
            image = image + np.float32(.125)
            yield "joint", i, {"image": image, "text": text}
        for i in range(20):
            image = image + np.float32(.125)
            yield "single", i, {"output": np.concatenate((text, image), axis=1)}
        yield "readout", None, {"noise": image}
        yield "commit", None, {"latent": latent + np.float32(1)}
    values = {"conditioning": np.ones((1, 1, 2), np.float32), "img_ids": np.zeros((2, 3), np.float32),
        "txt_ids": np.zeros((1, 3), np.float32), "timesteps": np.array([4, 3, 2, 1], np.float32),
        "sigmas": np.array([1, .75, .5, .25, 0], np.float32)}
    origin = {"inputs": {k: k for k in values}}
    class Reader:
        def array(self, key):
            if key in values:
                return values[key]
            kind, logical = key
            return np.array(values["timesteps"][logical] if kind == "timestep" else values["sigmas"][logical + (kind == "sigma_next")], np.float32)
    binding = {"pass_schedules": {str(s): {**{k: (k, s) for k in ("timestep", "sigma", "sigma_next")},
        "next_clock": {"step": s + 1, "phase": "project", "cursor": 0},
        "reset": {k: None for k in ("context", "image", "text", "noise_prediction")}} for s in (1, 2, 3)}}
    describe = lambda a: {"value": a.tolist(), "dtype": str(a.dtype)}
    rows = W["replay_suffix"](Reader(), binding, origin, np.zeros((1, 2, 2), np.float32),
        two={"independent_pass_events": events}, describe=describe, context_fields=lambda c: {"context.temb": c["temb"]},
        kind_at=kind_at, archive=lambda *a: {"kept": True}, output=tmp_path)
    assert len(rows) == 84
    assert [float(x[0, 0, 0]) for x in calls] == [0., 1., 2.]
    assert rows[-1]["outputs"]["latent"]["descriptor"]["value"][0][0][0] == 3
    assert sum(any(v["archive"] is not None for v in row["outputs"].values()) for row in rows) == 7
    assert len(W["BRANCHES"]) * len(rows) == 504
    assert 6 * sum(W["retained"](s, l) for s in (1, 2, 3) for l in range(28)) == 42


def test_cpu_command_is_explicit_and_model_free():
    launcher = runpy.run_path(str(Q / "submit_flux_latent_direction_cross.py"))
    request = {"config": {"cpu_threads": 4}, "execution_environment": launcher["L"]["B"]["execution_intent"]()}
    args = launcher["command"](request)
    assert "CUDA_VISIBLE_DEVICES=" in args
    assert "OPENBLAS_NUM_THREADS=4" in args
    assert args[-1] == "flux_latent_direction_cross.py"


def test_future_array_reader_is_restricted():
    h = W["helpers"]()
    class Evidence:
        def array(self, record):
            return record["value"]
    r = h["InputOnlyReader"](Evidence(), [{"value": 3}])
    assert r.array({"value": 3}) == 3
    with pytest.raises(ValueError, match="non-source"):
        r.array({"value": 7})


def test_duplicate_submission_rejected(tmp_path):
    launcher = runpy.run_path(str(Q / "submit_flux_latent_direction_cross.py"))
    (tmp_path / "submission.json").write_text("{}")
    with pytest.raises(ValueError, match="already admitted"):
        launcher["submit"](tmp_path)
