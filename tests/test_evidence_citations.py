from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from saturn_pub.evidence.citations import (
    CITATION_KEY,
    CITATION_VERSION,
    citation_fields,
    is_citation,
    main,
    make_citation,
    validate_address,
)

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "saturn_pub" / "evidence" / "citations.py"
ADDRESS = "ar://residual/layer/12/site/x"


def test_make_citation_minimal() -> None:
    citation = make_citation(ADDRESS)
    assert citation == {CITATION_KEY: CITATION_VERSION, "address": ADDRESS}


def test_make_citation_full_fields_and_canonical_json() -> None:
    citation = make_citation(
        ADDRESS,
        model_id="Example/Model-0.5B",
        revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
        input_sha256="a" * 64,
        quantity="bounded_sha256",
        note="prefill trace, probe 0",
    )
    assert citation[CITATION_KEY] == CITATION_VERSION
    assert citation["address"] == ADDRESS
    assert citation["model_id"] == "Example/Model-0.5B"
    assert citation["input_sha256"] == "a" * 64
    encoded = json.dumps(citation, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert json.loads(encoded) == citation


@pytest.mark.parametrize(
    "address",
    [
        None,
        123,
        "",
        "ar://",
        "ar:/residual/site/x",
        "http://residual/site/x",
        "ar://kv/layer/{layer}/key",
        "ar://residual/site/x.",
        "ar://residual/site/x/",
        "ar://residual site/x",
        "see ar://residual/site/x",
    ],
)
def test_make_citation_rejects_bad_addresses(address) -> None:
    with pytest.raises(ValueError):
        make_citation(address)


def test_validate_address_returns_input_unchanged() -> None:
    assert validate_address(ADDRESS) == ADDRESS


@pytest.mark.parametrize("input_sha256", ["", "xyz", "A" * 64, "a" * 63, "a" * 65, 42])
def test_make_citation_rejects_bad_input_sha256(input_sha256) -> None:
    with pytest.raises(ValueError, match="input_sha256"):
        make_citation(ADDRESS, input_sha256=input_sha256)


@pytest.mark.parametrize("field", ["model_id", "revision", "quantity", "note"])
@pytest.mark.parametrize("value", ["", "   ", 7, ["x"]])
def test_make_citation_rejects_bad_optional_strings(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        make_citation(ADDRESS, **{field: value})


def test_optional_fields_omitted_when_absent() -> None:
    citation = make_citation(ADDRESS, quantity="rms")
    assert set(citation) == {CITATION_KEY, "address", "quantity"}


def test_is_citation_marker_only() -> None:
    assert is_citation(make_citation(ADDRESS))
    assert is_citation({CITATION_KEY: CITATION_VERSION, "address": "not-checked-here"})
    assert not is_citation({CITATION_KEY: "v2", "address": ADDRESS})
    assert not is_citation({"address": ADDRESS})
    assert not is_citation("string")
    assert not is_citation(None)


def test_citation_fields_round_trip() -> None:
    citation = make_citation(
        ADDRESS, model_id="m", revision="r", input_sha256="b" * 64, quantity="q", note="n"
    )
    fields = citation_fields(citation)
    assert fields == {
        "address": ADDRESS,
        "model_id": "m",
        "revision": "r",
        "input_sha256": "b" * 64,
        "quantity": "q",
        "note": "n",
    }
    minimal = citation_fields(make_citation(ADDRESS))
    assert minimal["address"] == ADDRESS
    assert all(minimal[key] is None for key in ("model_id", "revision", "input_sha256"))


def test_citation_fields_fails_closed_never_raises() -> None:
    assert citation_fields(None) is None
    assert citation_fields([ADDRESS]) is None
    assert citation_fields({"address": ADDRESS}) is None
    assert citation_fields({CITATION_KEY: CITATION_VERSION}) is None
    assert citation_fields({CITATION_KEY: CITATION_VERSION, "address": "ar://x/{y}"}) is None
    fields = citation_fields(
        {
            CITATION_KEY: CITATION_VERSION,
            "address": ADDRESS,
            "model_id": 42,
            "revision": "",
            "input_sha256": "Z" * 64,
        }
    )
    assert fields is not None
    assert fields["model_id"] is None
    assert fields["revision"] is None
    assert fields["input_sha256"] is None


def test_cli_make_prints_canonical_json(capsys) -> None:
    assert main(["make", ADDRESS, "--model-id", "Example/Model-0.5B", "--quantity", "rms"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[CITATION_KEY] == CITATION_VERSION
    assert payload["model_id"] == "Example/Model-0.5B"


def test_cli_make_rejects_bad_address(capsys) -> None:
    assert main(["make", "ar://bad/{layer}"]) == 2
    assert "grammar" in capsys.readouterr().err


def test_module_is_stdlib_only_when_loaded_by_path(tmp_path: Path) -> None:
    """The writer-side helper must import without saturn_pub/torch (file-path load)."""

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('ec_standalone', {str(MODULE_PATH)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['ec_standalone'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        f"citation = mod.make_citation({ADDRESS!r}, quantity='rms')\n"
        "assert citation['saturn_pub_citation'] == 'v1'\n"
        "assert 'saturn_pub' not in sys.modules, 'saturn_pub leaked into stdlib-only module'\n"
        "assert 'torch' not in sys.modules, 'torch leaked into stdlib-only module'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path
    )
    assert completed.returncode == 0, completed.stderr


def test_grammar_matches_xref_extractor() -> None:
    """A valid citation address must index verbatim through the xref regex path."""

    from saturn_pub.evidence.xref import extract_addresses

    for address in (ADDRESS, "ar://kv/layer/003/position/00005/key"):
        make_citation(address)
        assert [found for found, _ in extract_addresses(f'"{address}"')] == [address]
