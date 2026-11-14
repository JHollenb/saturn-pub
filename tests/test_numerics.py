import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from saturn_pub.adapters.decoder import DecoderAdapter  # noqa: E402
from saturn_pub.core import Session  # noqa: E402
from saturn_pub.numerics import live_numeric_environment  # noqa: E402


@pytest.mark.parametrize("flag", ["bf16", "fp16"])
def test_reduced_precision_reduction_flag_binds_the_cut(flag):
    """The CUDA reduced-precision reduction flags are numeric-scope fields: a cut taken under
    one value is refused under the other, and a drifted live flag is named."""
    attribute = f"allow_{flag}_reduced_precision_reduction"
    field = f"{flag}_reduced_precision_reduction"
    matmul = torch.backends.cuda.matmul
    before = getattr(matmul, attribute)
    try:
        setattr(matmul, attribute, True)
        adapter = DecoderAdapter.tiny("qwen2")
        assert adapter.execution["numeric_scope"][field] is True
        assert live_numeric_environment()[field] is True
        session = adapter.session([1, 2, 3])
        session.continue_()
        cut = session.capture(retain=False)
        setattr(matmul, attribute, False)
        with pytest.raises(ValueError, match=f"{field}: recorded True, current False"):
            session.continue_()
        other = DecoderAdapter.tiny("qwen2")
        assert other.execution["numeric_scope"][field] is False
        with pytest.raises(ValueError, match=f"numeric_scope.{field}: cut True, adapter False"):
            Session.from_cut(other, cut)
        setattr(matmul, attribute, True)
        assert Session.from_cut(adapter, cut).capture(retain=False).fingerprint == cut.fingerprint
    finally:
        setattr(matmul, attribute, before)
