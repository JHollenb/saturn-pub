"""Finite Mamba VM page assays, distinct from a live bench session."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any
from .job_support import terminal_job, decode_marker, worker_command, stage_worker, guarded_mrun_launch, one_lease_scheduling_contract
from .paths import data_root
ROOT = data_root()
_VM_WORKER = Path(__file__).parent / "workers/run_saturn_mvm_op.py"
_VM_MARKER = "SATURN_MVM_OP_GZIP_B64="
_VM_MODES = ("read-formed-state", "residual-write", "forced-formed-install")
def _vm_build_config(op: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {'op': op, 'model': args.model, 'object': args.object, 'dtype': args.dtype}
    layers = getattr(args, 'layers', None)
    if layers:
        cfg['layers'] = [int(x) for x in str(layers).split(',') if x.strip() != '']
    if getattr(args, 'mode', None):
        cfg['mode'] = args.mode
    if getattr(args, 'donor', None):
        cfg['donor'] = args.donor
    return cfg

def _vm_request(op: str, cfg: dict[str, Any]) -> dict[str, Any]:
    from .execution_environment import intent
    return {'schema': 'saturn-mvm-op-mrun-request-v1', 'experiment': f'mvm-vm-{op}', 'task_family': 'forward', 'kind': 'vm-op', 'op': op, 'worker_sha256': hashlib.sha256(_VM_WORKER.read_bytes()).hexdigest(), 'device': 'cuda', 'needs': {'cuda': True}, 'run_config': cfg, 'execution_environment': intent(), 'scheduling_contract': one_lease_scheduling_contract(checkpoint_policy='one model load; one VM page op (mount/fork/swap/diff/unmount/sweep/clock) in process; no training, no nested submissions')}

def _vm_collect(op: str, job_id: str) -> int:
    (jid, record, log) = terminal_job(job_id, wait=True)
    report = decode_marker(log, _VM_MARKER)
    print(json.dumps({'op': op, 'job_id': jid, 'state': record.get('state'), 'report': report}, indent=2, sort_keys=True))
    return 0 if report is not None else 2

def _vm_submit(op: str, args: argparse.Namespace) -> int:
    if getattr(args, 'collect', None):
        return _vm_collect(op, args.collect)
    cfg = _vm_build_config(op, args)
    request = _vm_request(op, cfg)
    command = worker_command(_VM_WORKER.name, include_observability_extras=False)
    if args.dry_run:
        print(json.dumps({'op': op, 'command_tail': command[-3:], 'request': request, 'config': cfg}, indent=2, sort_keys=True))
        return 0
    import os as _os
    _os.environ.setdefault('MRUN_PREFLIGHT_DIR', str(ROOT / 'preflight-receipts'))
    with tempfile.TemporaryDirectory(prefix='saturn-mvm-cli-') as tmp:
        staging = Path(tmp)
        stage_worker(staging, worker=_VM_WORKER, config=cfg, request=request)
        job_id = guarded_mrun_launch(command, request=request, experiment=request['experiment'], task='forward', model=args.model, payload=staging, prefer_host='beast', ram_mb=args.ram, vram_mb=args.vram, gpu=True, mrun_config=request, timeout_s=1200.0, priority=100, detach=True, retry_on_kill=False, preflight=True, note=f'saturn-debug vm {op} - priority 100, do not preempt or cancel others')
    print(json.dumps({'op': op, 'job_id': str(job_id), 'experiment': request['experiment']}, sort_keys=True))
    return 0

def _vm_add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--model', default='mamba-130m', help='mrun model id or local path')
    parser.add_argument('--object', default='identity', choices=['identity', 'counting'])
    parser.add_argument('--dtype', default='fp32')
    parser.add_argument('--ram', type=float, default=None)
    parser.add_argument('--vram', type=float, default=None, help='measured peak x1.3')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--collect', metavar='JOB_ID', default=None, help='collect a submitted op')

def _add_vm_group(sub) -> None:
    vm = sub.add_parser('vm', help='Mamba model-virtual-memory page ops (through mrun)')
    vm_sub = vm.add_subparsers(dest='vm_command', required=True)
    for op in ('mount', 'fork', 'unmount', 'clock'):
        child = vm_sub.add_parser(op, help=f'{op} a Mamba state page set')
        _vm_add_common(child)
        if op == 'clock':
            child.add_argument('--donor', default='generic', choices=['generic', 'target'], help='reference page for write-magnitude deltas')
        child.set_defaults(handler=lambda args, op=op: _vm_submit(op, args))
    swap = vm_sub.add_parser('swap', help='per-layer page swap (read / residual-write / forced install)')
    _vm_add_common(swap)
    swap.add_argument('--layers', default='', help='comma-separated layer indices')
    swap.add_argument('--mode', default='read-formed-state', choices=_VM_MODES)
    swap.add_argument('--donor', default='generic', choices=['generic', 'target'])
    swap.set_defaults(handler=lambda args: _vm_submit('swap', args))
    diff = vm_sub.add_parser('diff', help='per-layer (C_l,H_l) diff of two programs')
    _vm_add_common(diff)
    diff.add_argument('--donor', default='generic', choices=['generic', 'target'])
    diff.set_defaults(handler=lambda args: _vm_submit('diff', args))
    sweep = vm_sub.add_parser('sweep', help='per-layer page sweep (reading or residual-writing)')
    _vm_add_common(sweep)
    sweep.add_argument('--mode', default='read-formed-state', choices=['read-formed-state', 'residual-write'])
    sweep.add_argument('--donor', default='generic', choices=['generic', 'target'])
    sweep.set_defaults(handler=lambda args: _vm_submit('sweep', args))

def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb vm", description=__doc__)
    # Reuse the complete original VM group and its parameter contracts.
    subs = parser.add_subparsers(dest="group", required=True)
    _add_vm_group(subs)
    args = parser.parse_args(["vm", *(argv or [])])
    return args.handler(args)
