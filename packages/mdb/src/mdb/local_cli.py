"""Local debugger entry points and dependency-only installation diagnostics."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from .debugger_render import cheat_sheet
from .errors import BenchError


def token_ids(raw):
    import argparse

    try:
        values = (
            json.loads(raw) if raw.strip().startswith("[") else [int(v) for v in raw.split(",")]
        )
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 0 for v in values)
        ):
            raise ValueError("expected a nonempty list of nonnegative integers")
        return values
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError("token IDs must be '1,2,3' or '[1,2,3]'") from exc


def add_local_options(parser):
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--tiny",
        nargs="?",
        const="counter",
        choices=("counter", "qwen2"),
        help="local demo: Counter needs no dependencies; Qwen uses random CPU weights",
    )
    selection.add_argument(
        "--local", type=Path, help="explicit local HF checkpoint directory; no downloads"
    )
    parser.add_argument(
        "--root", type=Path, help="native session directory; defaults to a new temporary path"
    )
    parser.add_argument("--resume", action="store_true", help="resume the native session at --root")
    parser.add_argument("--prompt", help="prompt for a locally loaded model with tokenizer files")
    parser.add_argument("--token-ids", type=token_ids, help="explicit input IDs, e.g. '1,2,3'")
    parser.add_argument("--device", default="cpu", help="explicit local device (default: cpu)")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32", "fp16"),
        help="Qwen requires bf16; Mamba defaults to fp32",
    )
    parser.add_argument("--seed", type=int, default=113, help="tiny Qwen initialization seed")
    parser.add_argument(
        "--max-steps", type=int, default=1024, help="finite native transitions per motion command"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="finite continue budget (Counter steps or generated tokens)",
    )
    parser.add_argument(
        "--export", type=Path, help="export a native session after successful commands"
    )


def run_local(args):
    from .debugger_cli import DebugShell
    from .local_debugger import connect_local, load_checkpoint, tiny_qwen

    if args.job_id or args.latest:
        raise BenchError("choose local debugging or job attachment")
    if args.endpoint:
        raise BenchError("--endpoint applies to job attachment; local debugging needs no server")
    if not 1 <= args.max_steps <= 100000 or not 1 <= args.max_new_tokens <= 4096:
        raise BenchError("max-steps must be 1..100000; max-new-tokens must be 1..4096")
    if not 0 <= args.seed < 2**63:
        raise BenchError("seed must be in 0..2**63-1")
    if args.prompt is not None and args.token_ids is not None:
        raise BenchError("choose --prompt or --token-ids")
    counter = args.local is None and args.tiny in {None, "counter"}
    bench = owner = None
    if counter:
        if args.device != "cpu":
            raise BenchError("Counter is a pure-Python demo; it does not use an accelerator")
        if args.root or args.resume or args.export or args.prompt or args.token_ids or args.dtype:
            raise BenchError(
                "Counter demo uses in-memory integer state; use --record to save its debugger receipt"
            )
        from .counter_debugger import CounterDebuggerAdapter

        adapter = CounterDebuggerAdapter()
        identity = {"model_id": "counter-plus-one", "scope": "pure-Python integer demo"}
        banner = "MDB Counter demo: pure Python; no trained model; cuts are in memory."
    else:
        from .bench import BenchSession
        from .bench_debugger import BenchDebuggerAdapter

        root = args.root.expanduser().resolve() if args.root else None
        if args.resume and (root is None or not (root / "selected.json").is_file()):
            raise BenchError("--resume needs --root pointing to an existing native session")
        if not args.resume and root and (root / "selected.json").exists():
            raise BenchError("session already exists; use --resume or a new --root")
        if args.resume and (args.prompt is not None or args.token_ids is not None):
            raise BenchError("resume uses retained history; do not supply new input")
        if args.local is None:
            if args.device != "cpu" or args.prompt is not None:
                raise BenchError(
                    "tiny Qwen uses random CPU weights and token IDs, not a text tokenizer"
                )
            model = tiny_qwen(seed=args.seed, dtype=args.dtype)
            tokenizer, family = None, "qwen2"
            model_id = f"tiny-qwen2-seed-{args.seed}"
        else:
            loader_ids = args.token_ids
            if args.resume:
                from .circuit_runtime import DirectoryArtifactStore

                selected = json.loads((root / "selected.json").read_text())
                manifest = DirectoryArtifactStore(root / "artifacts").get(selected["manifest"])
                binding = manifest["identity"]["tokenizer_sha256"]
                if binding in {"token-ids-only", "pipeline-owned-or-token-ids"}:
                    loader_ids = []  # Retain the original token-ID-only identity on resume.
            model, tokenizer, family = load_checkpoint(
                args.local, device=args.device, dtype=args.dtype, token_ids=loader_ids
            )
            model_id = str(args.local.expanduser().resolve())
        root = root or Path(tempfile.mkdtemp(prefix="mdb-local-")) / "session"
        try:
            bench = (
                BenchSession.resume(model, tokenizer, root=root)
                if args.resume
                else BenchSession.from_resident(
                    model,
                    tokenizer,
                    root=root,
                    model_id=model_id,
                    continuation_capacity=max(128, args.max_new_tokens),
                    resident_kv_bytes=64 << 20,
                )
            )
            if not args.resume:
                prompt = (
                    args.token_ids
                    if args.token_ids is not None
                    else ([1, 2, 3] if args.local is None else args.prompt or "Explain gravity.")
                )
                bench.prefill(prompt)
        except BaseException:
            if bench is not None:
                bench.close()
            raise
        adapter, identity = BenchDebuggerAdapter(bench), bench.identity
        banner = (
            "MDB tiny Qwen2: random CPU BF16 model; mechanics demo. "
            if args.local is None
            else f"MDB local {family}: "
        ) + f"session={root}"
    try:
        owner, session, motion = connect_local(
            adapter,
            identity=identity,
            bench=bench,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
        )
        if args.session_dir:
            session._session_dir = Path(args.session_dir).expanduser().resolve()
        shell = DebugShell(
            session, json_output=args.json, full_output=args.full, counter_demo=counter
        )
        if not args.json:
            print(banner)
        if args.command_line or args.script:
            if not args.json:
                print(cheat_sheet(counter))
                shell.emit_frame()
            ok = (
                shell.execute(args.command_line)
                if args.command_line
                else shell.run_script(args.script)["ok"]
            )
            result = 0 if ok else 1
        else:
            result = shell.repl()
        if result == 0 and args.export:
            shell.emit(session.call("export", {"destination": str(args.export)}))
        if args.record:
            session.write_receipt(args.record)
        return result
    finally:
        if owner is not None:
            owner.close()
        if bench is not None:
            bench.close()


def doctor_report(endpoint=None, token=None):
    versions = {}
    for distribution in ("mdb-runtime", "torch", "transformers", "transformer-lens", "mrun-pub"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    result = {
        "schema": "mdb-local-doctor-v1",
        "ok": True,
        "python": platform.python_version(),
        "versions": versions,
        "counter_demo": "available; no model dependencies",
        "native_qwen": "BF16 required; use --dtype bf16 or model.to(torch.bfloat16)",
        "scheduler": {"checked": False},
        "next": ["mdb debug", "mdb debug --tiny qwen2", "mdb help"],
    }
    if endpoint:
        request = Request(endpoint.rstrip("/") + "/health", headers={"Accept": "application/json"})
        if token:
            request.add_header("Authorization", "Bearer " + token)
        try:
            with urlopen(request, timeout=2) as response:
                healthy = 200 <= response.status < 300
            result["scheduler"] = {"checked": True, "reachable": healthy}
            result["ok"] = healthy
        except OSError as exc:
            # Never echo authorization headers or tokens in local diagnostics.
            result["scheduler"] = {"checked": True, "reachable": False, "error": type(exc).__name__}
            result["ok"] = False
    return result


def run_doctor(args):
    from .debugger_cli import _attach

    if args.job_id:
        return _attach(args)
    report = doctor_report(args.endpoint, args.token)
    if args.json or args.full:
        print(json.dumps(report, sort_keys=True, indent=None if args.json else 2))
    else:
        print(f"MDB local install: ready (Python {report['python']})")
        print("Counter: available without Torch or mrun")
        print(
            "Optional packages: "
            + ", ".join(
                f"{key}={value or 'not installed'}" for key, value in report["versions"].items()
            )
        )
        print(report["native_qwen"])
        print(
            "Scheduler: "
            + (
                "not checked (optional --endpoint URL)"
                if not args.endpoint
                else "reachable"
                if report["ok"]
                else "unreachable"
            )
        )
        print("Start: mdb debug | mdb debug --tiny qwen2 | mdb help")
    return 0 if report["ok"] else 1
