"""Exactness, probe-consistency and rewind checks for port stepping, with per-operation timings.

    python -m mdb.engine.selfcheck --tiny qwen2|qwen3 [--dtype bf16|fp32] --tokens N --json OUT
    python -m mdb.engine.selfcheck --model PATH [--dtype bf16|fp32] [--device cpu|cuda] \
        [--prompt TEXT | --token-ids 1,2,3] --tokens N --json OUT

Works on any registered Qwen2/Qwen3 checkpoint; nothing assumes tiny sizes. Numbers are
measured in this process unless the report's ``labels`` map says otherwise (the expected layer
count is derived-exact from the layer count and token count; ``generated`` and
``prompt_tokens`` are supplied).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .models import DTYPES, load_local, tiny_model  # noqa: E402,F401


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(samples: dict[str, list[float]], name: str, device: torch.device, fn, *args):
    _sync(device)
    start = time.perf_counter()
    result = fn(*args)
    _sync(device)
    samples.setdefault(name, []).append(time.perf_counter() - start)
    return result


def _summary(samples: dict[str, list[float]]) -> dict[str, Any]:
    return {
        name: {
            "count": len(values),
            "median_ms": statistics.median(values) * 1e3,
            "min_ms": min(values) * 1e3,
            "max_ms": max(values) * 1e3,
            "total_ms": sum(values) * 1e3,
        }
        for name, values in samples.items()
    }


def _digest(value: torch.Tensor) -> str:
    from .ports import tensor_digest

    return tensor_digest(value)


def check_exactness(adapter: Any, tokens: list[int], count: int, samples: dict) -> dict:
    """Same Session program with an observe-all controller vs none: receipts and cuts equal."""
    from .ports import PortController, program_for

    program = program_for(adapter)
    plain = adapter.session(tokens)
    observed = adapter.session(tokens)
    controller = PortController(program, None)
    logits_equal, cuts_equal, transitions = True, True, 0
    while plain.read("tokens").shape[1] < len(tokens) + count:
        _timed(samples, "transition_plain", adapter.device, plain.continue_)
        _timed(
            samples,
            "transition_observed",
            adapter.device,
            lambda: observed.continue_(1, ports=controller),
        )
        transitions += 1
        left, right = plain.capture(retain=False), observed.capture(retain=False)
        cuts_equal &= left.fingerprint == right.fingerprint
        if left.payload["logits"] is not None:
            logits_equal &= bool(
                torch.equal(
                    left.payload["logits"].view(torch.uint8),
                    right.payload["logits"].view(torch.uint8),
                )
            )
    receipts_equal = [r._body for r in plain.receipts] == [r._body for r in observed.receipts]
    generated = plain.read("tokens")[0].tolist()
    native = adapter.native_logits(generated[:-1])
    stepped = plain.read("logits")
    return {
        "transitions": transitions,
        "observed_layer_calls": controller.installs,
        "expected_layer_calls": adapter.layers * count,
        "receipts_byte_identical": receipts_equal,
        "cut_fingerprints_identical": cuts_equal,
        "logits_byte_identical": logits_equal,
        "stepped_vs_hf_full_forward_max_abs_logit_diff": float(
            (stepped.float() - native.float()).abs().max()
        ),
        "generated_tokens": generated[len(tokens) :],
    }


def check_probes(adapter: Any, tokens: list[int], samples: dict) -> dict:
    """Every port of every layer of the first row: stop-probe equals the one-pass capture."""
    from .ports import PortController, program_for

    program = program_for(adapter)
    session = adapter.session(tokens)
    session.continue_()  # embed
    mismatches, probes, s_checks, mix_checks, state_changes = [], 0, [], [], 0
    full_installs = 0
    for layer in range(adapter.layers):
        before = session.capture(retain=False)
        kv = [t.clone() for t in before.payload["keys"] + before.payload["values"]]
        full = PortController(program, layer)
        _timed(samples, "probe_full_layer", adapter.device, session.probe, full)
        full_installs += full.installs
        s_checks.append(full.checks.get("s"))
        mix_checks.append(full.checks.get("mix_replica"))
        for port in program.names:
            if port == "h_out":
                continue
            stop = PortController(program, layer, stop=port, capture=(port,))
            result = _timed(samples, "probe_stop", adapter.device, session.probe, stop)
            probes += 1
            if not result["stopped"] or not torch.equal(stop.captures[port], full.captures[port]):
                mismatches.append(f"L{layer}.{port}")
        after = session.capture(retain=False)
        state_changes += after.fingerprint != before.fingerprint
        payload = after.payload
        state_changes += not all(
            torch.equal(a, b) for a, b in zip(kv, payload["keys"] + payload["values"])
        )
        _timed(samples, "layer_commit_plain", adapter.device, session.continue_)
    return {
        "stop_probes": probes,
        "mismatches": mismatches,
        "state_changed_by_probe": state_changes,
        "receipts_after": len(session.receipts),
        "attempts_after": len(session.attempts),
        "history_after": len(session.history),
        "full_probe_installs": full_installs,
        "s_replica": s_checks,
        "mix_replica": mix_checks,
    }


def check_rewind(adapter: Any, tokens: list[int], samples: dict) -> dict:
    """Checkpoint mid-layer, step 7 ports, restore, step 7 again; also across a token edge."""
    from .run import Run

    run = Run(adapter, tokens)
    names = run.program.names
    results = {}
    last = adapter.layers - 1
    for case, target in (
        ("mid_layer", f"L0.{names[len(names) // 2]}"),
        ("token_boundary", f"L{last}.{names[-3]}"),
    ):
        _timed(samples, "until", adapter.device, run.until, target)
        checkpoint = _timed(samples, "checkpoint", adapter.device, run.checkpoint, case)
        first, second = [], []
        for trace in (first, second):
            for _ in range(7):
                _timed(samples, "port_step", adapter.device, run.step, "port")
                trace.append((run.where, _digest(run.read())))
            if trace is first:
                _timed(samples, "restore", adapter.device, run.restore, checkpoint)
        results[case] = {
            "checkpoint_where": checkpoint["where"],
            "checkpoint_bytes": len(json.dumps(checkpoint)),
            "path": [where for where, _ in first],
            "identical": first == second,
        }
        run.step("token")
    reverse = []
    for _ in range(3):
        _timed(samples, "reverse_port_step", adapter.device, run.reverse_step, "port")
        reverse.append(run.where)
    results["reverse_steps"] = reverse
    return results


def time_token(adapter: Any, tokens: list[int], samples: dict) -> None:
    from .run import Run

    run = Run(adapter, tokens)
    for _ in range(2):
        _timed(samples, "token_step_run", adapter.device, run.step, "token")
    session = adapter.session(tokens)
    _timed(samples, "token_generate_plain", adapter.device, adapter.generate, session, 1)


def environment(adapter: Any) -> dict[str, Any]:
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(adapter.device),
        "dtype": str(adapter.dtype),
        "attention_implementation": adapter.model.config._attn_implementation,
        "family": adapter.spec.model_type,
        "layers": adapter.layers,
        "hidden_size": adapter.hidden_size,
        "heads": adapter.num_heads,
        "kv_heads": adapter.num_kv_heads,
        "torch_num_threads": torch.get_num_threads(),
        "model_identity": adapter.model_identity,
        "cuda": torch.cuda.get_device_name(adapter.device)
        if adapter.device.type == "cuda"
        else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mdb.engine.selfcheck")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", choices=("qwen2", "qwen3"))
    source.add_argument("--model", help="local Hugging Face checkpoint directory")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=2, help="tokens to generate")
    parser.add_argument("--prompt", help="prompt text (needs the checkpoint tokenizer)")
    parser.add_argument("--token-ids", help="comma-separated prompt token ids")
    parser.add_argument("--json", required=True, help="output path")
    args = parser.parse_args(argv)
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    from saturn_pub.adapters.decoder import DecoderAdapter

    started = time.perf_counter()
    if args.tiny:
        model = tiny_model(args.tiny, dtype=args.dtype)
        adapter = DecoderAdapter(model, granularity="operation", device=torch.device(args.device))
    else:
        path = str(Path(args.model).expanduser())
        model = load_local(path, dtype=args.dtype, device=args.device)
        adapter = DecoderAdapter(model, granularity="operation", device=torch.device(args.device))
    load_seconds = time.perf_counter() - started
    prompt_encoding = "token ids"
    if args.token_ids:
        tokens = [int(item) for item in args.token_ids.split(",")]
    elif args.prompt and args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokens = tokenizer(args.prompt)["input_ids"]
        prompt_encoding = "checkpoint tokenizer"
    elif args.prompt:
        # a tiny random model has no tokenizer: byte values folded into its vocabulary
        tokens = [byte % adapter.vocab_size for byte in args.prompt.encode("utf-8")]
        prompt_encoding = "utf-8 bytes modulo vocab_size (tiny model has no tokenizer)"
    else:
        tokens = [1, 2, 3, 4, 5]
        prompt_encoding = "default ids"
    samples: dict[str, list[float]] = {}
    report: dict[str, Any] = {
        "schema": "mdb-engine-selfcheck-v1",
        "source": args.tiny and f"tiny:{args.tiny}" or "checkpoint",
        "prompt_tokens": len(tokens),
        "prompt_encoding": prompt_encoding,
        "generated": args.tokens,
        "environment": environment(adapter),
        "load_seconds": load_seconds,
        "labels": {
            "default": "measured",
            "exactness.expected_layer_calls": "derived-exact",
            "generated": "supplied",
            "prompt_tokens": "supplied",
        },
    }
    report["exactness"] = check_exactness(adapter, tokens, args.tokens, samples)
    report["probe_consistency"] = check_probes(adapter, tokens, samples)
    report["rewind"] = check_rewind(adapter, tokens, samples)
    time_token(adapter, tokens, samples)
    report["timings"] = _summary(samples)
    report["passed"] = passed(report)
    Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"passed": report["passed"], "json": args.json}))
    return 0 if report["passed"] else 1


def passed(report: dict[str, Any]) -> bool:
    """Every check holds, every replica check passed, and the controller really installed."""
    exact = report["exactness"]
    probes = report["probe_consistency"]
    rewind = report["rewind"]
    layers = report["environment"]["layers"]
    replicas = probes["s_replica"] + probes["mix_replica"]
    return bool(
        exact["receipts_byte_identical"]
        and exact["cut_fingerprints_identical"]
        and exact["logits_byte_identical"]
        and not probes["mismatches"]
        and not probes["state_changed_by_probe"]
        and probes["attempts_after"] == 0
        and probes["history_after"] == 0
        and all(item["identical"] for key, item in rewind.items() if key != "reverse_steps")
        and exact["transitions"] > 0
        and exact["observed_layer_calls"] == exact["expected_layer_calls"]
        and probes["full_probe_installs"] == layers
        and len(replicas) == 2 * layers
        and all(item == {"equal": True} for item in replicas)
    )


if __name__ == "__main__":
    sys.exit(main())
