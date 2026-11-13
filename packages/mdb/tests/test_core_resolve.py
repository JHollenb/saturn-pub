"""``mdb.resolve`` without a model: the stored-blob tier, ``unresolved`` with what would resolve
it, and the refusals (digest mismatch, foreign origin, corrupt blob, unknown recipe)."""
import types

import pytest

torch = pytest.importorskip("torch")

from test_core_recipe import TOKENS, _checkpoint, _held, _sha  # noqa: E402

import mdb  # noqa: E402
from mdb.core import index as I  # noqa: E402
from mdb.core import recipe as recipes  # noqa: E402
from mdb.core.pointers import occurrence_pointer  # noqa: E402
from mdb.core.project import Project  # noqa: E402
from mdb.core.resolve import TIERS, resolve  # noqa: E402

VALUE = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32) / 7


@pytest.fixture
def project(tmp_path):
    project = Project(tmp_path / "p")
    for sid in ("s1", "b"):
        project.open_session(sid, model_identity="m" * 64, execution={})
    return project


def _saved(project, sid="s1"):
    recipe = recipes.from_checkpoint(_checkpoint(), tokens=TOKENS, layers=2, tensors=_held(),
                                     store=project.store)
    recipes.save(project, sid, recipe)
    return recipe


def _pointer(recipe, origin="s1", sha256=None):
    return occurrence_pointer(f"{origin}:t3.L1.u", program="stepped", shape=[1, 1, 32],
                              dtype="float32", sha256=sha256 or _sha(VALUE),
                              recipe=recipe["fingerprint"], event=2, cut="c" * 64)


def _keep(project, pointer, sid="s1"):
    blob = project.store.put_tensor(VALUE)["ref"]
    I.index_occurrence(project, sid, pointer["address"], layers=2, recipe_id=pointer["recipe"],
                       event_id=2, cut_id="c" * 64, blob_id=blob)
    return blob


def test_mdb_resolve_is_the_core_resolver():
    assert mdb.resolve is resolve and "resolve" in mdb.__all__


def test_a_stored_blob_resolves_without_a_model(project):
    pointer = _pointer(_saved(project))
    _keep(project, pointer)
    result = resolve(pointer, project=project.root)   # a path opens the project read-only
    assert (result.state, result.tier, result.authority) == ("resolved", "blob", "measured")
    assert torch.equal(result.value, VALUE) and result.sha256 == pointer["value"]["sha256"]
    assert [a["tier"] for a in result.attempts] == ["live", "blob"]
    assert resolve(pointer, project=project, tiers=("blob",)).state == "resolved"


def test_without_a_blob_or_a_model_the_pointer_is_unresolved_with_what_would_resolve_it(project):
    pointer = _pointer(_saved(project))
    result = resolve(pointer, project=project)
    assert result.state == "unresolved" and result.value is None and result.authority is None
    assert [a["state"] for a in result.attempts] == ["unresolved"] * len(TIERS)
    assert "attach the model with identity mmmm" in result.needs
    assert "no stored blob is indexed" in result.reason


def test_a_digest_that_differs_from_the_pointer_is_refused(project):
    recipe = _saved(project)
    _keep(project, _pointer(recipe))
    result = resolve(_pointer(recipe, sha256="0" * 64), project=project)
    assert result.state == "refused" and result.value is None
    assert "digest mismatch" in result.reason


def test_a_foreign_origin_is_refused_until_the_project_holds_a_recipe_for_it(project):
    recipe = _saved(project)
    _keep(project, _pointer(recipe))
    attached = types.SimpleNamespace(session_id="s1", project=project)   # no live model
    foreign = _pointer(recipe, origin="b")
    result = resolve(foreign, session=attached)
    assert result.state == "refused"
    assert "origin b: is not the attached session 's1'" in result.reason
    assert "no recipe for 'b'" in result.reason
    assert resolve(_pointer(recipe), session=attached).state == "resolved"   # own origin
    _saved(project, "b")   # the same content is stored once, under s1: still no recipe for b
    assert resolve(foreign, session=attached).state == "refused"
    other = {**_checkpoint(), "name": "b-side"}
    b_recipe = recipes.from_checkpoint(other, tokens=TOKENS, layers=2, tensors=_held(),
                                       store=project.store)
    recipes.save(project, "b", b_recipe)
    refused = resolve(foreign, session=attached)
    assert refused.state == "refused" and "is not a recipe for origin b:" in refused.reason
    on_b = _pointer(b_recipe, origin="b")
    assert resolve(on_b, session=attached).state == "unresolved"   # b, not s1, is read
    _keep(project, on_b, sid="b")
    assert resolve(on_b, session=attached).tier == "blob"


def test_a_corrupt_value_blob_is_refused_naming_it(project):
    pointer = _pointer(_saved(project))
    blob = _keep(project, pointer)
    path = project.store.root / "tensors" / f"{blob}.safetensors"
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    result = resolve(pointer, project=project)
    assert result.state == "refused" and f"blob {blob} is corrupt" in result.reason


def test_an_unknown_recipe_or_a_missing_project_is_refused(project):
    pointer = {**_pointer(_saved(project)), "recipe": "f" * 64}
    assert "recipe ffff" in resolve(pointer, project=project).reason
    assert "needs project=" in resolve(pointer).reason
    with pytest.raises(ValueError, match="subset"):
        resolve(pointer, project=project, tiers=("disk",))
