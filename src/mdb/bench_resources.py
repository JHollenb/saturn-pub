"""Measured exceptions to the ordinary mrun diffusion capture envelope."""
from dataclasses import replace
import math


def diffusion_resources(model, *, height, width, steps, state_budget_bytes):
    from mrun.diffusion.resources import estimate_diffusion_resources
    resident = estimate_diffusion_resources(model, task="capture", height=height,
        width=width, steps=steps, phase_cuda=True, capture_sites=True, resident=True)
    capture = estimate_diffusion_resources(model, task="capture", height=height,
        width=width, steps=steps, phase_cuda=True, capture_sites=True)
    profile = replace(resident, disk_gb=capture.disk_gb, cpu_threads=4)
    basis = {"policy": "resident phase-CUDA model envelope with retained capture disk",
             "phase_vram_budget_decimal_mb": None}
    if model == "illustrious-xl" and height == width == 512 and steps == 4 and state_budget_bytes <= 512 << 20:
        # One full produce case covers native and checkpoint consumers, cuts,
        # debugger, discovery, siblings, eviction, export and a fresh child.
        # Normalize raw scheduler VRAM conservatively as MiB. The two state
        # allowances cover a full page pool and the preceding mutable closure.
        raw_peak = 7146
        normalized_peak = max(raw_peak*(1 << 20)/1e6, 6784.28672)
        allowance = 2*512*(1 << 20)/1e6
        reservation = math.ceil((normalized_peak*1.3+allowance)/512)*512
        profile = replace(profile, ram_mb=24000, vram_mb=reservation)
        basis = {"policy": "workload-matched sizing plus bounded retained-state allowance",
            "sizing_job": "job-282335c2f7d2", "native_pilot_job": "job-ddfbbaca0c83",
            "raw_scheduler_peak_vram_mb": raw_peak, "raw_units": "conservatively normalized as MiB",
            "pilot_torch_peak_reserved_decimal_mb": 6784.28672,
            "normalized_peak_decimal_mb": normalized_peak, "state_allowance_decimal_mb": allowance,
            "formula": "ceil512(max(normalized mrun peak, pilot torch peak)*1.3 + 2*512 MiB)",
            "vram_reservation_mb": reservation, "mrun_kill_ceiling_mb": reservation*1.1,
            "phase_vram_budget_decimal_mb": reservation*1.1-40,
            "ram_basis": "24GB conservative retained archive/page-cache envelope; sizing RSS9890.4MB",
            "boundary": "derived reservation, not an observed six-case peak; one resident model; batch1;512px;4steps;state<=512MiB"}
    return profile, basis
