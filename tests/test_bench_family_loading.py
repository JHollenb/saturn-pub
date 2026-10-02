import pytest
from transformers import MambaConfig, Qwen2Config

from mdb.bench import BenchError, resident_family


def test_family_override_cannot_reinterpret_native_abi():
    class Resident:
        config = Qwen2Config()
    assert resident_family(Resident()) == "qwen2"
    with pytest.raises(BenchError, match="incompatible"):
        resident_family(Resident(), "mamba")
    Resident.config = MambaConfig()
    assert resident_family(Resident()) == "mamba"
    Resident.config.model_type = "mamba2"
    with pytest.raises(BenchError, match="unsupported"):
        resident_family(Resident())
