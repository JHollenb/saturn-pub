"""Tiny CPU mechanics example; no pretrained WSP or mathematical ability claim."""
from __future__ import annotations

import json
from pathlib import Path


def run_demo(root):
    import numpy as np
    import torch

    from .response_closure import measure_error
    from .wsp import WeightStudy, verify_weight_study
    from .wsp_weights import ParameterPatch, parameter_inventory, weight_diff

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    model = torch.nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
    before = parameter_inventory(model)
    replacement = torch.tensor([[0., 0.], [1., 1.]])
    patch = ParameterPatch(model, {"weight": replacement}, model_identity="tiny-policy-v1")
    baseline = model.weight.detach().numpy().copy()
    edited = replacement.numpy().copy()

    def trajectory(matrix, *, restore=False):
        state, states, actions = np.array([1., 0.]), [], []
        for _ in range(3):
            state = (baseline if restore else matrix) @ state
            states.append(state.copy())
            action = int(np.argmax(state))
            actions.append(action)
            # Action 1 is the supplied environment's terminal action.
            if action == 1:
                break
            state = np.eye(2)[action]
        return {"states": np.asarray(states), "actions": np.asarray(actions, dtype=np.int64),
                "completed": bool(actions[-1] == 1)}

    def native(context):
        restore = context.declaration["role"] == "restoration"
        state, states, actions = torch.tensor([1., 0.]), [], []
        for _ in range(3):
            output = context.model(state)
            # A local activation restoration control; the next invocation uses
            # this branch's selected action, never an observed reference future.
            if restore:
                output = torch.from_numpy(baseline) @ state
            states.append(output.detach().numpy().copy())
            action = int(output.argmax())
            actions.append(action)
            if action == 1:
                break
            state = torch.eye(2)[action]
        return {"observations": {"states": np.asarray(states),
                    "actions": np.asarray(actions, dtype=np.int64), "completed": actions[-1] == 1},
                "work": {"model_forwards": len(actions), "restoration_matvecs": len(actions) if restore else 0},
                "state": {"policy": "restart", "initial_state": [1., 0.],
                          "feedback": "own argmax action to one-hot input", "max_steps": 3}}

    rows = {}
    with WeightStudy(root / "study", model=model, binding={"model": "tiny-policy-v1",
        "checkpoint": before["identity"], "execution": "CPU FP32 Torch; NumPy FP64 forecast",
        "consumer": "supplied two-action toy environment", "horizon": "at most three own-action steps"},
        input_identity={"initial_state": [1., 0.]}, source_identity={"demo": "mdb.wsp_demo.v1"}) as study:
        arms = [("native", "native", None, baseline), ("edited", "edited", patch, edited),
                ("zero", "sham", patch.scaled(model, 0), baseline),
                ("restore", "restoration", patch, baseline)]
        for name, role, candidate, matrix in arms:
            study.add_arm(name, role=role, patch=candidate, state_policy="restart",
                context={"supplied_terminal_action": 1, "candidate_role_status": "toy construction"},
                predictor=lambda context, m=matrix: {"predictions": trajectory(m)})
        seal = study.seal()
        for name, *_ in arms:
            study.run(name, native)
            observed = study.tx.observation(name)["result"]["observations"]
            forecast = study.tx.prediction(name)
            rows[name] = {"actions": observed["actions"].tolist(), "completed": observed["completed"],
                          "state_forecast_error": measure_error(forecast["states"], observed["states"])}
        report = study.report()
    result = {"scope": "tiny CPU mechanics, supplied policy and environment; no pretrained model result",
              "rows": rows, "parameters_restored": parameter_inventory(model) == before,
              "diff": weight_diff({"weight": torch.from_numpy(baseline)}, {"weight": replacement}),
              "seal": seal, "verification": verify_weight_study(root / "study"), "study": report}
    with (root / "report.json").open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result
