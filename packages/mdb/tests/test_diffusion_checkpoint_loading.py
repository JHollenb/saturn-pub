"""Diffusers path resolution uses configured authorities across public clients."""
from types import SimpleNamespace

import mrun.models
import pytest

from mdb.bench_loading import local_checkpoint_root


@pytest.mark.parametrize("extra", [False, True])
def test_older_public_client_without_search_helper(tmp_path, monkeypatch, extra):
    selected = tmp_path / ("extra" if extra else "models") / "FLUX.2-klein-4B"
    selected.mkdir(parents=True)
    (selected / "model_index.json").write_text("{}")
    monkeypatch.delattr(mrun.models, "model_search_roots", raising=False)
    monkeypatch.setattr(mrun.models, "resolve_model", lambda name: SimpleNamespace(
        family="flux", local_path=None, hf_id="black-forest-labs/FLUX.2-klein-4B", name=name))
    monkeypatch.setattr(mrun.models, "models_root", lambda: tmp_path / "models")
    monkeypatch.setattr(mrun.models, "default_hub_root", lambda: tmp_path / "hub")
    monkeypatch.setenv("LLM_MODELS_EXTRA_ROOT", str(tmp_path / "extra") if extra else "")
    assert local_checkpoint_root("flux2-klein-4b-distilled") == selected
