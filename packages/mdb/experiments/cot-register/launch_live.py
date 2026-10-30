"""Stage one bounded native trial through mrun's intent-first admission."""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mdb.execution_environment import intent  # noqa: E402
from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit"))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--measured-admission-from", help="retained same-worker successful or VRAM-killed memory basis")
    parser.add_argument("--wheel", type=Path, help="standalone MDB wheel; never execute from the checkout")
    parser.add_argument("--lens", type=Path, help="retained, model-bound frozen Jacobian lens directory")
    parser.add_argument("--fast-comparison", action="store_true", help="three-arm fast/strict/native assay and selected restore")
    parser.add_argument("--footprint-admission-from", help="same-model completed footprint; changed worker/geometry explicitly recorded")
    parser.add_argument("--output-base", default="/mnt/big/mdb-cot-register", help="host-local artifact volume directory")
    args = parser.parse_args()
    stage = args.stage.resolve()
    if args.action == "prepare":
        stage.mkdir(parents=True, exist_ok=False)
        wheel = args.wheel or HERE / 'results/job-6ece44afdab3/payload/mdb_runtime-0.1.0-py3-none-any.whl'
        lens = args.lens or HERE / 'results/job-6ece44afdab3/payload/lens'
        if wheel.name != 'mdb_runtime-0.1.0-py3-none-any.whl':
            raise ValueError('this bootstrap expects the qualified mdb_runtime-0.1.0 wheel filename')
        for source in (HERE / 'bootstrap_live.py', HERE / 'live_trial.py', wheel):
            shutil.copyfile(source, stage / source.name)
        if args.fast_comparison:
            shutil.copyfile(HERE / 'live_trial.py', stage / 'trial_support.py')
            shutil.copyfile(HERE / 'fast_trial.py', stage / 'live_trial.py')
        shutil.copytree(lens, stage / 'lens')
        nonce = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        protocol = {"schema": "mdb-cot-register-live-protocol-v1", "model_id": "Qwen/Qwen2.5-1.5B",
            "model_path": "/mnt/big/llm-models/Qwen2.5-1.5B", "revision": "local-frozen-a961db72e75d52b1",
            "model_safetensors_sha256": "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796",
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "max_worker_seconds": 600,
            "output_root": str(Path(args.output_base) / ("trial-" + nonce)),
            "cases": [
                {"id": "france", "prompt": "The capital of France is", "max_new_tokens": 8},
                {"id": "city-role", "prompt": "The first city is Rome. The second city is Madrid. The country containing the second city is", "max_new_tokens": 8},
                {"id": "arithmetic", "prompt": "Question: What is 13 + 24?\nAnswer: Let's think step by step.", "max_new_tokens": 24},
                {"id": "animal-two-hop", "prompt": "Question: How many legs does the animal that spins a web have?\nAnswer: Let's think step by step.", "max_new_tokens": 24}]}
        if args.fast_comparison:
            protocol["schema"] = "mdb-cot-fast-protocol-v1"
            protocol["arms"] = ["native", "strict", "fast"]
            protocol["output_root"] = str(Path(args.output_base) / ("fast-" + nonce))
        (stage / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        inventory = {str(path.relative_to(stage)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in stage.rglob('*') if path.is_file()}
        (stage / 'payload-manifest.json').write_text(json.dumps(inventory, sort_keys=True, indent=2) + '\n')
        request = {"schema": "mdb-cot-register-live-request-v1", "experiment": "mdb-cot-register-live",
            "task_family": "decode", "model": protocol["model_id"], "backend": "hf",
            "instrument_backend": "mdb-native-eager", "device": "cuda", "dtype": "bfloat16",
            "batch_size": 1, "seq_len": 128, "needs": {"cuda": True},
            "backend_options": {"physical_batch_size": 1, "full_prefill_logits": True},
            "source_sha256": inventory, "output_root": protocol["output_root"],
            "execution_environment": intent(), "retry_on_kill": False,
            "workload_geometry": {"model_loads": 1, "scalar_prefills": 9,
                "max_token_commits": 136, "max_layer_advances": 3808, "backward_calls": 0,
                "physical_batch_size": 1, "full_vocabulary_observer": True},
            "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_branches=9,
                logical_operations=136, logical_checkpoint_replays=0, max_local_work_items=256,
                checkpoint_policy="one loaded Qwen; sequential own-history native and observed arms; no fitting, nested jobs, or retries"),
            "scheduler": {"strict_preflight": True, "retry_on_kill": False}}
        if args.fast_comparison:
            request["experiment"] = "mdb-cot-register-fast"
            request["workload_geometry"].update(scalar_prefills=13, max_token_commits=205,
                max_layer_advances=5740, selected_checkpoint_replays=1)
            request["scheduling_contract"] = one_lease_scheduling_contract(model_loads=1,
                logical_branches=13, logical_operations=205, logical_checkpoint_replays=1,
                max_local_work_items=256, checkpoint_policy="one frozen load; rotated fast/strict/native arms; one selected Thott restore; no fitting or retries")
        if args.measured_admission_from and args.footprint_admission_from:
            raise ValueError("choose one admission basis")
        if args.footprint_admission_from:
            if not args.fast_comparison:
                raise ValueError("changed-worker footprint exception is scoped to the fast comparison")
            basis_path = HERE / 'results' / args.footprint_admission_from / 'job-record.json'
            basis = json.loads(basis_path.read_text())
            old = basis['config']
            if basis['state'] != 'succeeded':
                raise ValueError("footprint exception requires a completed retained run")
            for field in ('model', 'backend', 'device', 'dtype', 'seq_len', 'batch_size'):
                if old[field] != request[field]:
                    raise ValueError("model footprint differs: " + field)
            for name in ('lens/manifest.json', 'lens/tensors.safetensors'):
                if old['source_sha256'][name] != inventory[name]:
                    raise ValueError("reader geometry differs: " + name)
            peak = float(basis['result']['peak_vram_mb'])
            request['measured_admission'] = {'basis_job': basis['job_id'],
                'basis_record_sha256': hashlib.sha256(basis_path.read_bytes()).hexdigest(),
                'measured_peak_vram_mb': peak, 'factor': 1.3, 'requested_vram_mb': peak * 1.3,
                'requested_ram_mb': float(basis['reservation']['ram_mb']),
                'ram_basis': 'completed same-model worker original RAM grant; paired RAM ask required to honor VRAM ask',
                'operational_change': 'different wheel/worker; 13 histories versus 9, and one selected restore; physical batch/context/capacity and model unchanged; one resident 4.5MiB transport, no head duplicate',
                'qualification': 'bounded same-model footprint exception, not a same-worker measurement; 30% headroom and strict RAM/VRAM guards remain active',
                'prior_workload_geometry': old['workload_geometry'], 'new_workload_geometry': request['workload_geometry']}
        if args.measured_admission_from:
            basis_path = HERE / 'results' / args.measured_admission_from / 'job-record.json'
            basis = json.loads(basis_path.read_text())
            old = basis['config']
            if basis['state'] not in {'succeeded', 'killed_vram'} or basis['experiment'] != request['experiment']:
                raise ValueError('measured admission requires this experiment\'s retained successful or VRAM-killed run')
            for field in ('model', 'backend', 'device', 'dtype', 'workload_geometry'):
                if old[field] != request[field]:
                    raise ValueError('measured admission geometry differs: ' + field)
            for name in ('live_trial.py', 'bootstrap_live.py', wheel.name,
                         'lens/manifest.json', 'lens/tensors.safetensors'):
                if old['source_sha256'][name] != inventory[name]:
                    raise ValueError('measured admission worker/asset identity differs: ' + name)
            peak = float(basis['result']['peak_vram_mb'])
            request['measured_admission'] = {'basis_job': basis['job_id'],
                'basis_record_sha256': hashlib.sha256(basis_path.read_bytes()).hexdigest(),
                'measured_peak_vram_mb': peak, 'factor': 1.3,
                'requested_vram_mb': peak * 1.3,
                'requested_ram_mb': float(basis['reservation']['ram_mb']),
                'ram_basis': 'retain the same worker geometry\'s original mrun-planned RAM grant; mrun currently requires RAM to honor a declared VRAM ask',
                'censoring': ('resource-killed load peak is a lower bound; full-run peak remains unknown'
                             if basis['state'] == 'killed_vram' else
                             'completed same-worker peak; next attempt remains protected by mrun resource guards'),
                'operational_change': 'fresh output destination; identical worker, assets and physical geometry'}
        (stage.parent / 'request.json').write_text(json.dumps(request, indent=2, sort_keys=True) + '\n')
        print(json.dumps({"stage": str(stage), "request": request}, sort_keys=True))
        return
    request = json.loads((stage.parent / 'request.json').read_text())
    command = ['/usr/bin/env', 'PYTHONUNBUFFERED=1', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=',
        'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'TOKENIZERS_PARALLELISM=false',
        'SATURN_RESIDENT_WORKER=1', '/mnt/big/llm-models/.uv-cache/archive-v0/Z8hg7fARI9lZbMS7/bin/python',
        '-I', '-u', 'bootstrap_live.py']
    job = guarded_mrun_launch(command, request=request, experiment=request['experiment'], task='decode',
        model=request['model'], payload=stage, prefer_host='beast', gpu=True, needs=request['needs'],
        mrun_config=request, timeout_s=840, detach=True, retry_on_kill=False, preflight=True,
        ram_mb=request.get('measured_admission', {}).get('requested_ram_mb'),
        vram_mb=request.get('measured_admission', {}).get('requested_vram_mb'),
        note=("MDB fast CoT assay: one frozen Qwen1.5B BF16 CUDA load; retained L12 lens; four rotated native/strict/fast triples and one selected Thott restore; <=205 scalar commits; no fitting/retries; 600s stop and raw rows/views retained."
              if request['experiment'] == 'mdb-cot-register-fast' else
              "Live MDB CoT trial: one frozen Qwen1.5B BF16 CUDA load; retained L12 lens, four observed/native pairs, <=136 scalar token commits, no fitting; stop after 600s and retain views/raw rows."))
    (stage.parent / 'job.json').write_text(json.dumps({"job_id": job, "request": request}, indent=2) + '\n')
    print(json.dumps({"job_id": job, "output_root": request['output_root']}, sort_keys=True))


if __name__ == '__main__':
    main()
