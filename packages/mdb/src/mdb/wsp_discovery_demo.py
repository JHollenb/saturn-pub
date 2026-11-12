"""Blinded constructed-organism discovery; no pretrained or learned claim."""
from __future__ import annotations

import json
from pathlib import Path


def run_discovery_demo(root, *, seed=731):
    import numpy as np
    import torch

    from .wsp import WeightStudy, verify_weight_study
    from .wsp_discovery import AffineDiscovery, virtual_edit
    from .wsp_weights import parameter_inventory

    rng = np.random.default_rng(seed)

    class Organism(torch.nn.Module):
        def __init__(self):
            super().__init__()
            names = ["site_" + str(i) for i in rng.permutation(5)]
            self.parts = torch.nn.ModuleDict({name: torch.nn.Linear(4, 4, bias=False,
                                                                  dtype=torch.float64) for name in names})
            with torch.no_grad():
                for module in self.parts.values():
                    u, v = rng.normal(size=(2, 4))
                    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
                    module.weight.copy_(torch.tensor(.1 * np.outer(u, v)))

        def forward(self, value):
            for module in self.parts.values():
                value = value + module(value)
            return value

    model = Organism().eval().requires_grad_(False)
    names = list(model.parts)
    fault_index = int(rng.integers(0, len(names)))
    fault_name = "parts." + names[fault_index]
    # Oracle construction remains in this fixture. Only examples and an output
    # evaluator cross the controller API; no healthy weights or fault labels do.
    task = torch.tensor(rng.normal(size=(32, 4)))
    held = torch.tensor(rng.normal(size=(48, 4)))
    healthy_task, healthy_held = model(task).detach(), model(held).detach()
    prefix = np.eye(4)
    for module in list(model.parts.values())[:fault_index]:
        prefix = (np.eye(4) + module.weight.detach().numpy()) @ prefix
    weight = list(model.parts.values())[fault_index].weight.detach().numpy()
    _, _, vh = np.linalg.svd(weight)
    trigger = vh[0] @ prefix
    preserve = rng.normal(size=(24, 4))
    preserve -= np.outer(preserve @ trigger / (trigger @ trigger), trigger)
    preserve = torch.tensor(preserve)
    with torch.no_grad():
        list(model.parts.values())[fault_index].weight.mul_(8)
    preservation_target = model(preserve).detach()
    original = parameter_inventory(model)
    identity = {"organism": "constructed-residual-v1", "parameters": original["identity"]}
    binding = {"model": identity, "checkpoint": original["identity"], "execution": "CPU FP64",
               "consumer": "four-dimensional function; task MSE and preservation MSE",
               "horizon": "five residual writes"}
    root = Path(root)

    def evaluator(net):
        prediction, kept = net(task), net(preserve)
        task_error = float((prediction - healthy_task).square().mean())
        preserve_error = float((kept - preservation_target).square().mean())
        return {"observations": {"objective": -task_error - preserve_error,
                    "task_mse": task_error, "preservation_mse": preserve_error},
                "work": {"physical_forwards": 2, "task_rows": len(task), "preservation_rows": len(preserve)},
                "state": {"policy": "restart", "recurrent_state": False}}

    controller = AffineDiscovery(model, root, binding=binding,
        input_identity={"discovery_seed": seed, "task_rows": len(task), "preservation_rows": len(preserve)},
        source_identity={"fixture": "blind-constructed-residual-v1"}, max_sites=32, shortlist=2,
        channels_per_site=1)
    chosen, patch, selection = controller.discover(evaluator)
    if patch is None:
        return selection
    module = dict(model.named_modules())[chosen.site]
    old_weight = module.weight.detach().clone()

    def heldout(net):
        return {"observations": {"outputs": net(held), "task_mse": float((net(held) - healthy_held).square().mean())},
                "work": {"physical_forwards": 2, "rows": len(held)},
                "state": {"policy": "restart", "recurrent_state": False}}

    final_root = root / "heldout"
    with WeightStudy(final_root, model=model, binding=binding,
        input_identity={"heldout_rows": len(held), "heldout_inputs_sha256":
                        __import__("hashlib").sha256(held.numpy().tobytes()).hexdigest()},
        source_identity={"selected_from": "discovery examples only"}) as study:
        for arm, role, candidate in (("native", "native", None), ("virtual", "edited", None),
                ("compiled", "edited", patch), ("restored", "restoration", patch)):
            study.add_arm(arm, role=role, patch=candidate, state_policy="restart",
                predictor=lambda context, name=arm: {"predictions": {
                    "expectation": "virtual and compiled implement the same affine map; restoration returns native",
                    "arm": name, "selected_edit": selection["selected"]}})
        study.seal()
        for arm in ("native", "virtual", "compiled", "restored"):
            def run(context, arm=arm):
                if arm == "virtual":
                    with virtual_edit(context.model, chosen):
                        return heldout(context.model)
                if arm == "restored":
                    hook = module.register_forward_hook(lambda m, args, output:
                        torch.nn.functional.linear(args[0], old_weight))
                    try:
                        return heldout(context.model)
                    finally:
                        hook.remove()
                return heldout(context.model)
            study.run(arm, run)
        outcomes = {arm: study.tx.observation(arm)["result"]["observations"]
                    for arm in ("native", "virtual", "compiled", "restored")}
    report = {"schema": "mdb-blind-affine-discovery-demo-v1", "scope": "constructed blind mechanics organism",
        "seed": seed, "selected": selection["selected"], "hidden_fault_site": fault_name,
        "hidden_fault_gain": 8., "site_discovered": chosen.site == fault_name,
        "discovery_candidates": len(selection["records"]),
        "heldout_mse": {name: row["task_mse"] for name, row in outcomes.items()},
        "virtual_compiled_max_abs": float(np.abs(outcomes["virtual"]["outputs"] - outcomes["compiled"]["outputs"]).max()),
        "restored_native_max_abs": float(np.abs(outcomes["restored"]["outputs"] - outcomes["native"]["outputs"]).max()),
        "parameter_restoration": parameter_inventory(model) == original,
        "verification": verify_weight_study(final_root)}
    with (root / "demo-report.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report
