from dataclasses import dataclass

from mdb.bench_resources import diffusion_resources


def test_measured_envelope_is_limited_to_qualified_geometry(monkeypatch):
    from mrun.diffusion import resources

    @dataclass
    class Profile:
        ram_mb: int = 52000
        vram_mb: int = 14000
        disk_gb: int = 25
        cpu_threads: int = 1

    monkeypatch.setattr(resources, "estimate_diffusion_resources", lambda *a, **k: Profile())
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
