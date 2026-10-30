"""Qualify one longitudinal symbol trace on a resident pretrained decoder."""

import base64
import faulthandler
import gzip
import hashlib
import io
import json
import os
import runpy
import tarfile
import time
from pathlib import Path


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from mrun.models import snapshot_dir
    from saturn_pub.adapters.decoder import DecoderAdapter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    started = time.monotonic()
    torch.manual_seed(113)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model_path = snapshot_dir(config["model"])
    print("SYMBOL_TRACE_PHASE=load", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, device_map="cuda", attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
    )
    token_ids = tokenizer.encode(config["prompt"])
    if len(token_ids) > config["max_prompt_tokens"]:
        raise ValueError("tokenized prompt exceeds the admitted geometry")
    # An explicitly staged assay driver is an executable payload, not a sibling checkout.
    qualify = runpy.run_path(str(Path(__file__).with_name("run_demo.py")))["qualify"]
    root = Path("symbol-trace-output").resolve()
    print("SYMBOL_TRACE_PHASE=workflow", flush=True)
    faulthandler.dump_traceback_later(45, repeat=True)
    try:
        result = qualify(DecoderAdapter(model), token_ids, root)
    finally:
        faulthandler.cancel_dump_traceback_later()
    counts = result["physical_work"]
    geometry = request["geometry"]
    if (counts["native_transitions"] > geometry["native_transition_limit"]
            or counts["interventions"] != geometry["interventions"]
            or counts["bindings"] < 2):
        raise RuntimeError("workflow differs from its admitted geometry")
    result["decoded_next_tokens"] = {
        arm: tokenizer.decode([result["controls"][arm]["logit_argmax"]])
        for arm in ("main", "copy", "zero")
    }
    result.update(
        schema="mdb-symbol-trace-qualification-v1", job_id=os.environ["MRUN_JOB_ID"],
        model=config["model"], device="cuda", dtype="float32", attention="eager",
        prompt=config["prompt"], prompt_tokens=len(token_ids),
        elapsed_seconds=time.monotonic() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        mechanics_status="valid", instrument_status="native-workflow-exercised",
        trend_status="not-assessed", terminal_status="not-assessed",
        claim_boundary="One pretrained native workflow; supplied symbol names; no semantic discovery.",
        request_sha256=request["request_sha256"],
        execution_environment=json.loads(Path("execution-environment.json").read_text()),
    )
    (root / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(root, arcname="symbol-trace-output")
    raw = buffer.getvalue()
    if len(raw) > 8 * 1024 * 1024:
        raise RuntimeError("project archive exceeds the finite 8 MiB collection budget")
    result["archive"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    encoded = base64.b64encode(gzip.compress(json.dumps(result, sort_keys=True).encode())).decode()
    print("SYMBOL_TRACE_REPORT_V1 " + encoded, flush=True)
    print("SYMBOL_TRACE_ARCHIVE_V1 " + base64.b64encode(raw).decode("ascii"), flush=True)


if __name__ == "__main__":
    main()
