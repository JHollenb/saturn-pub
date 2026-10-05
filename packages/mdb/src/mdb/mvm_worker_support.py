"""Shared model-virtual-memory op dispatch for the Mamba VM.

One code path serves both the ``saturn-debug vm ...`` CLI (which stages a thin
worker that calls :func:`run_mamba_op`) and the validation smoke worker, so the
CLI prints the same evidence the experiment records.  Model loading lives here
(lease-side, CUDA by default); the VM mechanics live in :mod:`saturn.mamba_state_vm`
and import cleanly on the laptop for unit tests.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import torch

from mdb.mamba_state_vm import MambaStateVM, build_task
from mdb.model_os_mamba_state import OP_READ, OP_RESIDUAL_WRITE


# The E3b full-layer-sweep prompt objects, verbatim, so VM sweeps reproduce the
# recorded sweep130 profile.
OBJECTS: dict[str, dict[str, Any]] = {
    "identity": {
        "prompts": {
            "base": "a photorealistic fox sitting in fresh snow at dawn, soft light. The animal in this scene is a",
            "generic": "a photorealistic creature sitting in fresh snow at dawn, soft light. The animal in this scene is a",
            "target": "a photorealistic lion sitting in fresh snow at dawn, soft light. The animal in this scene is a",
        },
        "carriers": {"base": ["fox"], "generic": ["creature"], "target": ["lion"]},
        "answer_base": "fox", "answer_target": "lion",
        "distractors": ["wolf", "tiger", "cat", "dog", "deer", "bear", "rabbit"],
    },
    "counting": {
        "prompts": {
            "base": "Anna put three apples in the basket. The number of apples in the basket is",
            "generic": "Anna put some apples in the basket. The number of apples in the basket is",
            "target": "Anna put five apples in the basket. The number of apples in the basket is",
        },
        "carriers": {"base": ["three"], "generic": ["some"], "target": ["five"]},
        "answer_base": "three", "answer_target": "five",
        "distractors": ["one", "two", "four", "six", "seven", "nine"],
    },
}


def model_abi_digest(model_name: str, dtype: torch.dtype) -> str:
    return hashlib.sha256(f"saturn-mvm:{model_name}:{dtype}".encode()).hexdigest()


def load_causal_lm(model_name: str, dtype: torch.dtype, *, device: str = "cuda"):
    """Load a stock HF causal LM (local path or mrun model id), identity returned."""

    if "/" in model_name or os.path.isdir(model_name):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
        ).to(device).eval()
        return model, tok, model_name, "local-path"
    from mrun.models import load_hf_model, load_tokenizer, resolve_model

    spec = resolve_model(model_name)
    tok = load_tokenizer(spec.name, local_files_only=True)
    model = load_hf_model(spec, device_map={"": device}, dtype=dtype, local_files_only=True).eval()
    return model, tok, spec.hf_id, "local-materialized"


def build_vm_for_object(model_name: str, *, dtype: torch.dtype, object_name: str, device: str = "cuda"):
    """Load the model, build the Mamba VM, mount the object's base program.

    Returns ``(vm, tok, task, ident)`` where ``ident`` describes the loaded model.
    The base program is mounted under session ``"base"``; donor programs
    (``generic``/``target``) are returned in ``task["programs"]``.
    """

    model, tok, model_id, revision = load_causal_lm(model_name, dtype, device=device)
    device_resolved = str(next(model.parameters()).device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    abi = model_abi_digest(model_name, dtype)
    vm = MambaStateVM(model, model_abi_sha256=abi, model_id=model_id, tokenizer_identity=f"{model_id}:tok")
    task = build_task(vm, tok, OBJECTS[object_name])
    vm.mount(
        "base", task["programs"]["base"], suffix=task["suffix"], span_ids=task["span_ids"],
        cut_pos=task["cut_pos"], positions=task["positions"], readout=task["readout"],
    )
    ident = {
        "model": model_name, "model_id": model_id, "revision": revision,
        "device": device_resolved, "dtype": str(dtype), "architecture": type(model).__name__,
        "execution_abi": vm.runtime.execution_abi_sha256, "object": object_name,
        "n_layers": vm.runtime.n_layers,
    }
    return vm, tok, task, ident


def run_mamba_op(config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one VM page op from an mrun run-config and return a receipt report.

    ``config`` keys: ``model``, ``object`` (identity/counting), ``op`` (one of
    mount/fork/swap/diff/unmount/sweep/clock/gates), ``dtype``, plus op params
    (``layers``, ``mode``, ``donor``).  ``donor`` selects the registered program
    (``generic`` or ``target``).
    """

    dtype = torch.float32 if str(config.get("dtype", "fp32")).lower() in ("fp32", "float32") else torch.bfloat16
    object_name = str(config.get("object", "identity"))
    op = str(config["op"])
    vm, _tok, task, ident = build_vm_for_object(
        str(config["model"]), dtype=dtype, object_name=object_name, device=str(config.get("device", "cuda")),
    )
    programs = task["programs"]
    donor = programs.get(str(config.get("donor", "generic")), programs["generic"])
    report: dict[str, Any] = {"schema": "saturn-mvm-op-v1", "op": op, "ident": ident}

    if op == "mount":
        report["result"] = vm.mount(
            "remount", programs["base"], suffix=task["suffix"], span_ids=task["span_ids"],
            cut_pos=task["cut_pos"], positions=task["positions"], readout=task["readout"],
        )
    elif op == "fork":
        report["result"] = vm.fork("base", "base-fork")
    elif op == "gates":
        report["result"] = vm.gates("base")
    elif op == "swap":
        report["result"] = vm.swap(
            "base", donor=donor, layers=[int(x) for x in config.get("layers", [ident["n_layers"] - 1])],
            mode=str(config.get("mode", OP_READ)),
        )
    elif op == "diff":
        report["result"] = vm.diff(programs["base"], donor)
    elif op == "unmount":
        report["result"] = vm.unmount("base")
    elif op == "clock":
        report["result"] = vm.clock("base", reference=donor)
    elif op == "sweep":
        mode = str(config.get("mode", OP_READ))
        if mode == OP_RESIDUAL_WRITE:
            report["result"] = vm.sweep("base", donor=programs["base"], target=programs["target"], mode=mode)
        else:
            report["result"] = vm.sweep("base", donor=donor, mode=mode)
    else:
        raise ValueError(f"unknown vm op {op!r}")
    return report
