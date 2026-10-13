from dataclasses import dataclass
import sys
from types import ModuleType

from mdb.bench_resources import diffusion_resources


def test_measured_envelope_is_limited_to_qualified_geometry(monkeypatch):
    # The estimator is already supplied by this fixture. Keep the optional
    # private mrun distribution out of the public mechanics test environment.
    resources = ModuleType("mrun.diffusion.resources")
    diffusion = ModuleType("mrun.diffusion")
    diffusion.resources = resources
    mrun = ModuleType("mrun")
    mrun.diffusion = diffusion
    monkeypatch.setitem(sys.modules, "mrun", mrun)
    monkeypatch.setitem(sys.modules, "mrun.diffusion", diffusion)
    monkeypatch.setitem(sys.modules, "mrun.diffusion.resources", resources)

    @dataclass
    class Profile:
        ram_mb: int = 52000
        vram_mb: int = 14000
        disk_gb: int = 25
        cpu_threads: int = 1

    resources.estimate_diffusion_resources = lambda *a, **k: Profile()
    measured, basis = diffusion_resources("illustrious-xl", height=512, width=512,
        steps=4, state_budget_bytes=512 << 20)
    assert measured.vram_mb == 11264 and measured.ram_mb == 24000
    assert basis["phase_vram_budget_decimal_mb"] < basis["mrun_kill_ceiling_mb"]
    for changed in ({"height": 1024}, {"steps": 5}, {"state_budget_bytes": 513 << 20},
                    {"model": "flux2-klein-4b-distilled"}):
        config = dict(model="illustrious-xl", height=512, width=512, steps=4,
            state_budget_bytes=512 << 20)
        config.update(changed)
        profile, fallback = diffusion_resources(**config)
        assert profile.vram_mb == 14000 and profile.ram_mb == 52000
        assert profile.cpu_threads == 4 and fallback["phase_vram_budget_decimal_mb"] is None
