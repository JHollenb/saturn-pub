"""Shared mechanics for a blinded, paired editor-workflow calibration."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ARMS = ("wsp_python", "pytorch_hooks", "mdb")
BLOCKED_ATTRIBUTES = {"load", "save", "loadtxt", "savetxt", "fromfile", "tofile", "memmap", "ctypeslib",
    "storage", "untyped_storage", "set_default_device", "set_num_threads", "compile", "jit",
    "forward", "train", "requires_grad_", "register_parameter", "register_buffer", "load_state_dict",
    "set_submodule", "add_module"}


def public_getattr(value, name, *default):
    """Public numerical introspection under the same policy as dotted access."""
    if not isinstance(name, str) or name.startswith("_") or name in BLOCKED_ATTRIBUTES:
        raise ValueError("private or disallowed numerical attribute")
    return getattr(value, name, *default)


def public_hasattr(value, name):
    try:
        public_getattr(value, name)
        return True
    except AttributeError:
        return False


def plain(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def fixture(seed, kind, *, include_evaluation=True):
    """Constructed, untrained specimen. Hidden oracle never enters editor input."""
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    names = ["site_" + str(i) for i in rng.permutation(5)]

    class Organism(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parts = torch.nn.ModuleDict({n: torch.nn.Linear(4, 4, bias=False, dtype=torch.float64) for n in names})

        def forward(self, value):
            for module in self.parts.values():
                value = value + module(torch.tanh(value))
            return value

    healthy = {n: rng.normal(size=(4, 4)) * rng.uniform(.035, .12) for n in names}
    broken = {n: w.copy() for n, w in healthy.items()}
    index = int(rng.integers(0, len(names)))
    gain = {"amplification": 4.3, "sign": -1.6, "healthy": 1.}[kind]
    broken[names[index]] *= gain
    model = Organism().eval().requires_grad_(False)

    def set_weights(values):
        with torch.no_grad():
            for name, matrix in values.items():
                model.parts[name].weight.copy_(torch.tensor(matrix, dtype=torch.float64))

    # Generate each split independently. Held arrays and oracle are absent from
    # the numerical code environment, the serialized public packet and history.
    splits = {}
    for split, count, shift in (("discovery", 8, 0.), ("preservation", 8, 1.),
                                ("evaluation", 48, 0.), ("fresh_preservation", 48, 1.)):
        if not include_evaluation and split in {"evaluation", "fresh_preservation"}:
            continue
        x = torch.tensor(rng.normal(size=(count, 4)) + shift, dtype=torch.float64)
        set_weights(healthy)
        with torch.inference_mode():
            y = model(x).clone()
        splits[split] = (x, y)
    set_weights(broken)
    return model, splits, {"kind": kind, "fault_site": "parts." + names[index] if gain != 1 else None,
        "fault_gain": gain, "healthy_weights": {"parts." + n + ".weight": w.tolist() for n, w in healthy.items()}}


def validate_edits(edits, sites):
    import math
    if not isinstance(edits, list) or len(edits) > 2:
        raise ValueError("edits must be a list of at most two affine write gains")
    result = []
    for row in edits:
        if not isinstance(row, dict) or set(row) != {"site", "gain"} or row["site"] not in sites:
            raise ValueError("unknown edit site or fields")
        gain = row["gain"]
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or not -4 <= gain <= 4:
            raise ValueError("gain must be finite and in [-4,4]")
        result.append({"site": row["site"], "gain": float(gain)})
    if len({r["site"] for r in result}) != len(result):
        raise ValueError("duplicate gain site")
    return sorted(result, key=lambda r: r["site"])


def validate_program(code):
    """Numerical experiment policy, not a general adversarial Python sandbox."""
    if not isinstance(code, str) or len(code) > 18000:
        raise ValueError("numerical program exceeds declared size")
    tree = ast.parse(code)
    forbidden = (ast.Import, ast.ImportFrom, ast.While, ast.Global, ast.Nonlocal, ast.ClassDef)
    def attribute_target(target):
        if isinstance(target, ast.Attribute):
            return True
        if isinstance(target, ast.Subscript):
            return attribute_target(target.value)
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(attribute_target(child) for child in target.elts)
        return False
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError("numerical program contains a disallowed statement")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("private numerical names are unavailable")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BLOCKED_ATTRIBUTES):
            raise ValueError("private or I/O numerical attributes are unavailable")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(attribute_target(target) for target in targets):
                raise ValueError("numerical programs cannot assign model/module attributes")
    return tree


def editor_schema():
    return {"type": "object", "properties": {
        "analysis_python": {"type": "string"}, "mechanism": {"type": "string"},
        "forecast": {"type": "string"}},
        "required": ["analysis_python", "mechanism", "forecast"], "additionalProperties": False}


COMMON = """You are the fixed experimental weight-editing policy. Return JSON only, without invoking any tools.
The supplied model is a constructed, untrained CPU FP64 residual program: for each ordered
torch.nn.Linear site, x <- x + site(torch.tanh(x)). Some specimens need no edit.
Repair the supplied discovery function while preserving the supplied preservation function.
No healthy weights, fault label or evaluation inputs are available. No autograd, backward calls,
optimizer state or training loops. Explicit analytic response equations and sensitivities are permitted.
Write analysis_python to run in the supplied numerical kernel. Available: np, torch, model,
x,y (8 discovery rows), keep_x,keep_y (8 preservation rows), sites (ordered names), weights
(copies of original matrices), history (your own prior round only), probe(edits), and print.
probe([{'site': name, 'gain': g}, ...]) evaluates a temporary parameter edit and returns
task_mse, preservation_mse, objective=-task_mse-preservation_mse and outputs. It restores the
original model. At most two distinct sites; finite gains in [-4,4]. Gain multiplies each site's
original weight. Direct permanent parameter mutation is invalid. All native model calls,
including taps and probes, spend the same six-call budget this round. There are two rounds.
You may use finite Python/NumPy/Torch numerical analysis, bounded loops, functions and hooks.
No imports, files, private attributes, while loops, backward, optimizers or network access.
Use only public attributes. Print a small diagnostic if useful. Programs have the cooperative
deadline stated in the public packet. Candidate outcomes persist even if a later operation raises.
The final selection is automatic: greatest observed discovery objective across both rounds,
including original; retain original on a tie. You cannot choose using evaluation data.
Your mechanism and forecast are recorded before this round executes; say unknown when needed.
Do not implement the complete native network again to evade its model-call budget. Small local
response equations and weight algebra are allowed and their host time is paid work.
"""

DOCS = {
    "wsp_python": """Workflow: WSP-style Python weight inspection and outcome feedback. All common Python,
Torch and hook capabilities remain available. Inspect weights, reason about the computation,
and propose numerical edits. No additional interpretation labels or repair procedure is supplied.
This is a bounded calibration of that workflow, not a reproduction of a publication's full harness.
""",
    "pytorch_hooks": """Workflow: competent ordinary Python/PyTorch inspection with hooks. You can capture
each site's actual input and return with this standard recipe (one native model call):
taps = {}
handles = []
for name, module in model.named_modules():
    if name in sites:
        def hook(module, args, out, name=name):
            taps[name] = (args[0].detach().clone(), out.detach().clone())
        handles.append(module.register_forward_hook(hook))
try:
    output = model(x)
finally:
    for handle in handles: handle.remove()
For an affine gain g, at a fixed incoming operand h the direct change is
(g-1) * (h @ weights[site].T). Later operands can change too. Compare full native outcomes
through probe; local response size alone need not predict the final task effect. Hooks can
be reused or narrowed. Parameter snapshots and try/finally restore are handled by probe.
""",
    "mdb": """Workflow: the common Python/PyTorch kernel plus installed public MDB.
capture(value=x) records one unchanged forward on the supplied inputs using MDB record_forward(capture_ports=True),
then resolves actual module input/return ports through NativePortCatalog. It returns a dict
by site with input, output, input_port and output_port; tensors are Torch FP64 copies.
linear_response(W, Wprime, h) is the public MDB FP64 equation returning NumPy arrays with
baseline, edited and direct. At a fixed operand, direct = h @ (Wprime-W).T. Later receivers
may change. Weight norms or local response size alone do not predict full consumer effects.
Your probe uses a ParameterPatch and a sealed WeightStudy before native evaluation, retaining
forecasts, candidate identities, outputs and restoration records. All root native calls and
capture calls spend the same budget as the other workflows. You also retain all raw Python
and hook capabilities. These tools supply execution facts, not semantic fault labels.
"""}
