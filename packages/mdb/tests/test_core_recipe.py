"""Replay recipe v2 without a model: the one-to-one map from ``Run.checkpoint()``, tensors held
as store-blob pointers (never inline), bounds, seal, and each distinct refusal."""
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from saturn_pub.store import LocalStore  # noqa: E402
from saturn_pub.values import digest  # noqa: E402

from mdb.core import index as I  # noqa: E402
from mdb.core import recipe as recipes  # noqa: E402
from mdb.core.errors import Refused  # noqa: E402
from mdb.core.pointers import RecordError  # noqa: E402
from mdb.core.project import Project  # noqa: E402

TOKENS = [1, 5, 9, 3]
SCOPE = {"torch_num_threads": 1, "deterministic_algorithms": False,
         "float32_matmul_precision": "highest", "tf32_matmul": False}


def _sha(value):
    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                          ).hexdigest()


def _tensors():
    value = torch.linspace(-1, 1, 32).reshape(1, 1, 32)
    mask = torch.zeros(1, 1, 16, dtype=torch.bool)
    mask[..., :4] = True
    other = torch.full((1, 1, 16), 1.5)
    return value, mask, other


def _checkpoint(tokens=TOKENS):
    value, mask, other = _tensors()
    tensor = {"kind": "tensor", "sha256": _sha(value), "shape": [1, 1, 32],
              "dtype": "torch.float32"}
    masked = {"port": "h_out", "op": "replace",
              "value": {"kind": "tensor", "sha256": _sha(other), "shape": [1, 1, 16],
                        "dtype": "torch.float32"},
              "selection": "first four", "mask_sha256": _sha(mask), "mask_count": 4,
              "mask_shape": [1, 1, 16]}
    return {
        "schema": recipes.CHECKPOINT_SCHEMA, "name": "pre", "model_identity": "m" * 64,
        "prompt_sha256": digest(tokens), "where": "t3.L1.sg", "cut": "c" * 64, "transitions": 2,
        "port": "sg", "lookahead": "none", "numeric_scope": dict(SCOPE),
        "edits": [{"port": "u", "op": "add", "value": {"kind": "scalar", "value": 0.125}}],
        "path_edits": [{"transition": 1, "layer": 0, "edits": [
            {"port": "a", "op": "scale", "value": {"kind": "scalar", "value": 0.5}},
            {"port": "u", "op": "add", "value": tensor}, masked]}],
    }


def _held():
    return {_sha(t): t for t in _tensors()}


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path / "store")


def _recipe(store, checkpoint=None, tokens=TOKENS):
    return recipes.from_checkpoint(checkpoint or _checkpoint(tokens), tokens=tokens, layers=2,
                                   tensors=_held(), store=store)


def test_a_recipe_is_the_checkpoint_one_to_one_with_tensors_as_blob_pointers(store):
    checkpoint = _checkpoint()
    recipe = _recipe(store)
    assert recipes.to_checkpoint(recipe) == checkpoint
    assert recipe["fingerprint"] == I.recipe_fingerprint(
        {k: v for k, v in recipe.items() if k != "fingerprint"})
    pointers = recipes.tensor_pointers(recipe)
    assert [p["kind"] for p in pointers] == ["tensor"] * 3
    assert {p["sha256"] for p in pointers} == set(_held())
    assert recipe["staged"][0]["value"] == {"kind": "scalar",
                                            "value": {"value": 0.125, "authority": "supplied"}}
    text = json.dumps(recipe)
    assert "-1.0" not in text and "linspace" not in text   # no tensor data inline
    loaded = recipes.load_tensors(recipe, store)
    assert set(loaded) == set(_held())
    for sha, tensor in _held().items():
        assert torch.equal(loaded[sha], tensor) and loaded[sha].dtype == tensor.dtype


def test_long_prompts_are_held_by_pointer(store):
    tokens = list(range(100))
    recipe = _recipe(store, tokens=tokens)
    assert recipe["tokens"]["kind"] == "tensor" and recipe["tokens"]["dtype"] == "int64"
    assert recipes.load_tokens(recipe, store) == tokens
    assert recipes.load_tokens(_recipe(store), store) == TOKENS


def test_an_oversized_recipe_or_one_holding_an_array_is_refused(store, tmp_path):
    checkpoint = _checkpoint()
    checkpoint["path_edits"] = [{"transition": n, "layer": 0, "edits": []} for n in range(65)]
    with pytest.raises(RecordError, match="store it by pointer"):
        _recipe(store, checkpoint)
    body = {k: v for k, v in _recipe(store).items() if k != "fingerprint"}
    with pytest.raises(RecordError, match="array or tensor"):
        recipes.seal({**body, "name": torch.zeros(2)})
    with pytest.raises(RecordError, match="bytes"):
        recipes.seal({**body, "target": {**body["target"], "where": ["x" * 4000] * 17}})
    project = Project(tmp_path / "p")
    project.open_session("s1", model_identity="m" * 64, execution={})
    with pytest.raises(recipes.RecipeError, match="malformed replay recipe"):
        recipes.save(project, "s1", {**body, "fingerprint": "0" * 64})
    assert I.recipe_ids(project, "s1") == []


def test_a_tampered_recipe_is_refused(store):
    recipe = _recipe(store)
    for changed in ({**recipe, "cut": "d" * 64}, {**recipe, "lookahead": "layer"},
                    {k: v for k, v in recipe.items() if k != "name"}):
        with pytest.raises(recipes.RecipeError, match="malformed replay recipe") as caught:
            recipes.verify(changed)
        assert caught.value.code == "malformed"
    staged = [{**recipe["staged"][0], "value": {"kind": "scalar", "value": 0.125}}]
    with pytest.raises(recipes.RecipeError, match="num"):
        recipes.seal({**recipe, "staged": staged})


def test_missing_and_corrupt_blobs_are_refusals_naming_the_blob(store):
    recipe = _recipe(store)
    victim = recipes.tensor_pointers(recipe)[0]["ref"]
    path = store.root / "tensors" / f"{victim}.safetensors"
    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(recipes.RecipeError, match=f"blob {victim}.*corrupt") as caught:
        recipes.load_tensors(recipe, store)
    assert caught.value.code == "corrupt-blob" and isinstance(caught.value, Refused)
    path.unlink()
    with pytest.raises(recipes.RecipeError, match=f"blob {victim}.*missing") as caught:
        recipes.load_tensors(recipe, store)
    assert caught.value.code == "missing-blob"


def test_binding_refusals_are_distinct(store):
    recipe = _recipe(store)
    recipes.check_binding(recipe, model_identity="m" * 64, numeric_scope=SCOPE)
    with pytest.raises(recipes.RecipeError, match="wrong model identity") as caught:
        recipes.check_binding(recipe, model_identity="n" * 64, numeric_scope=SCOPE)
    assert caught.value.code == "model-identity"
    with pytest.raises(recipes.RecipeError,
                       match="torch_num_threads: recipe 1, this process 2") as caught:
        recipes.check_binding(recipe, model_identity="m" * 64,
                              numeric_scope={**SCOPE, "torch_num_threads": 2})
    assert caught.value.code == "numeric-scope"
    mismatch = recipes.fingerprint_mismatch(recipe, "c" * 64, "e" * 64)
    assert mismatch.code == "fingerprint" and "fingerprint mismatch" in str(mismatch)


def test_occurrence_checkpoints_cut_the_history_at_the_event(store):
    recipe = _recipe(store)
    inside = recipes.occurrence_checkpoint(recipe, where="t3.L0.u", cut="b" * 64, event=1,
                                           port="u")
    assert inside["path_edits"] == [] and [e["port"] for e in inside["edits"]] == ["a", "u", "h_out"]
    assert (inside["cut"], inside["transitions"], inside["port"]) == ("b" * 64, 1, "u")
    boundary = recipes.occurrence_checkpoint(recipe, where="t3.L0.h_out", cut="b" * 64, event=2,
                                             port=None)
    assert boundary["edits"] == [] and len(boundary["path_edits"]) == 1
    at_target = recipes.occurrence_checkpoint(recipe, where="t3.L1.sg", cut="c" * 64, event=2,
                                              port="sg")
    assert at_target["edits"] == _checkpoint()["edits"]
    with pytest.raises(recipes.RecipeError, match="past the recipe target"):
        recipes.occurrence_checkpoint(recipe, where="t4.emb", cut="c" * 64, event=9, port=None)


def test_recipes_are_saved_and_loaded_through_the_project_index(store, tmp_path):
    project = Project(tmp_path / "p")
    project.open_session("s1", model_identity="m" * 64, execution={})
    recipe = recipes.from_checkpoint(_checkpoint(), tokens=TOKENS, layers=2, tensors=_held(),
                                     store=project.store)
    identifier = recipes.save(project, "s1", recipe)
    assert identifier == recipe["fingerprint"] and recipes.load(project, identifier) == recipe
    record = recipes.record_checkpoint(project, "s1", recipe)
    assert I.get_cursor(project, "s1") == record["id"]
    assert recipes.find_checkpoint(project, "pre", session_id="s1")["id"] == record["id"]
    assert recipes.find_checkpoint(project, "pre")["id"] == record["id"]
    assert recipes.find_checkpoint(project, "other", session_id="s1") is None
