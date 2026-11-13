"""Sparse state/read predictors trained with public Saturn training transactions."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attention_ranking import evaluate_route_forecast
from .native_discovery import (
    DiscoveryError,
    checked,
    error_metrics,
    read_arrays,
    save_arrays,
    save_json,
    sealed,
)

KINDS = ("linear_features", "sparse_reconstruction", "sparse_consumer")
# Position handling of scales, initial write and loss normalisation.
#   "global": one input centre/scale and one write scale for every position (the original runs).
#   "split":  separate centre/scale and write scale for position 0 and for positions 1..; the
#             initial write equals the position-split mean write; each row's state error is
#             normalised by its own class scale, so a massive first-token vector cannot dominate.
POSITION_MODES = ("global", "split")
# Development continuation selection.
#   "original":        development output KL (when a consumer is supplied), then own-depth
#                      normalised MSE, then local consumer loss.
#   "position_robust": development output KL, then own-depth centred relative MSE on positions
#                      1.. (position 0 excluded), then local consumer loss.
SELECTIONS = ("original", "position_robust")
ROBUST_SUFFIX = ".p0robust"


def arm_spec(value):
    """Normalise one learned-arm specification (a kind name or a dict) with explicit defaults."""
    spec = {"kind": value} if isinstance(value, str) else dict(value)
    unknown = set(spec) - {"name", "kind", "position_mode", "selection", "read_groups"}
    if unknown:
        raise DiscoveryError("unknown arm option: " + ", ".join(sorted(unknown)))
    spec.setdefault("name", spec.get("kind"))
    spec.setdefault("position_mode", "global")
    spec.setdefault("selection", "original")
    spec.setdefault("read_groups", 1)
    if (spec["kind"] not in KINDS or spec["position_mode"] not in POSITION_MODES
            or spec["selection"] not in SELECTIONS
            or not (spec["read_groups"] == "kv" or (type(spec["read_groups"]) is int and spec["read_groups"] >= 1))
            or not isinstance(spec["name"], str)):
        raise DiscoveryError("invalid learned-arm specification: " + repr(value))
    return spec


def robust_arm(kind):
    """The position-0-robust configuration: split scales/initial write, robust selection, KV-group reads."""
    return {"name": kind + ROBUST_SUFFIX, "kind": kind, "position_mode": "split",
            "selection": "position_robust", "read_groups": "kv"}


ARM_PRESETS = {
    "original": tuple(arm_spec(k) for k in KINDS),
    "original+p0robust": tuple(arm_spec(k) for k in KINDS) + tuple(arm_spec(robust_arm(k)) for k in KINDS),
}


def resolve_arms(arms, *, kv_groups=None):
    """Arm specs from a preset name, a list of kinds/dicts or ``None`` (the original three)."""
    if arms is None:
        arms = "original"
    if isinstance(arms, str):
        if arms not in ARM_PRESETS:
            raise DiscoveryError("unknown arm preset: " + arms)
        arms = ARM_PRESETS[arms]
    specs = [arm_spec(a) for a in arms]
    if len({s["name"] for s in specs}) != len(specs) or not specs:
        raise DiscoveryError("learned arm names must be unique and nonempty")
    for spec in specs:
        if spec["read_groups"] == "kv":
            if not kv_groups:
                raise DiscoveryError("KV-group reads need the native key/value head count")
            spec["read_groups"] = int(kv_groups)
    return specs


def position_classes(length, device=None):
    """Row class per position: 0 for position 0, 1 for every later position."""
    return (torch.arange(length, device=device) > 0).long()


class SpanPredictor(nn.Module):
    def __init__(self, width, features, heads, value_width, *, kind, topk=8, rank=4,
                 position_mode="global", read_groups=1):
        super().__init__()
        if position_mode not in POSITION_MODES:
            raise DiscoveryError("unknown position mode")
        if heads % read_groups or value_width % read_groups:
            raise DiscoveryError("read groups must divide heads and value width")
        self.kind, self.heads, self.rank = kind, heads, rank
        self.position_mode, self.read_groups = position_mode, read_groups
        self.topk = min(topk, features)
        self.encoder = nn.Linear(width, features)
        self.query = nn.Linear(features, heads * rank, bias=False)
        self.key = nn.Linear(features, heads * rank, bias=False)
        self.value = nn.Linear(features, value_width)
        self.local_write = nn.Linear(features, width)
        self.read_write = nn.Linear(value_width, width, bias=False)
        classes = () if position_mode == "global" else (2,)
        self.register_buffer("center", torch.zeros(*classes, width))
        self.register_buffer("scale", torch.ones(classes))
        self.register_buffer("write_scale", torch.ones(classes))
        self.register_buffer("value_scale", torch.ones(()))
        self.register_buffer("value_center", torch.zeros(value_width))
        if position_mode == "split":
            # Fixed position-0 write offset so that the initial write equals the split mean write.
            self.register_buffer("first_shift", torch.zeros(width))
        nn.init.zeros_(self.local_write.weight)
        nn.init.zeros_(self.local_write.bias)
        nn.init.zeros_(self.read_write.weight)

    def row_write_scale(self, length, device=None):
        """Write scale per row, ``[T, 1]`` (a scalar for the global mode)."""
        if self.position_mode == "global":
            return self.write_scale
        return self.write_scale[position_classes(length, device)][:, None]

    def encode(self, x):
        if self.position_mode == "global":
            raw = self.encoder((x - self.center) / self.scale)
        else:
            rows = position_classes(x.shape[0], x.device)
            raw = self.encoder((x - self.center[rows]) / self.scale[rows][:, None])
        if self.kind == "linear_features":
            return raw
        positive = raw.relu()
        ids = positive.topk(self.topk, dim=-1).indices
        return torch.zeros_like(positive).scatter(-1, ids, positive.gather(-1, ids))

    def read(self, route, normalized_value):
        """Predicted read. One group averages every head (original); G groups keep KV groups apart."""
        if self.read_groups == 1:
            return route.mean(0) @ normalized_value
        g, t = self.read_groups, route.shape[-1]
        grouped = route.reshape(g, self.heads // g, t, t).mean(1)
        values = normalized_value.reshape(t, g, -1).transpose(0, 1)
        return (grouped @ values).transpose(0, 1).reshape(t, -1)

    def forward(self, x):
        code = self.encode(x)
        t = x.shape[0]
        q = self.query(code).reshape(t, self.heads, self.rank).transpose(0, 1)
        k = self.key(code).reshape(t, self.heads, self.rank).transpose(0, 1)
        score = q @ k.transpose(-1, -2) / self.rank ** 0.5
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        route = score.masked_fill(mask, -torch.inf).softmax(-1)
        value = self.value(code) * self.value_scale + self.value_center
        read = self.read(route, (value - self.value_center) / self.value_scale)
        write = (self.local_write(code) + self.read_write(read)) * self.row_write_scale(t, x.device)
        if self.position_mode == "split":
            write = write + self.first_shift * (position_classes(t, x.device) == 0)[:, None]
        return {"state": x + write, "routes": route, "values": value, "code": code}


class StatePredictor(nn.Module):
    def __init__(self, *, width, features, heads, value_width, stages, kind, topk=8, rank=4,
                 position_mode="global", read_groups=1):
        super().__init__()
        if (kind not in KINDS or position_mode not in POSITION_MODES
                or min(width, features, heads, value_width, stages, topk, rank, read_groups) < 1):
            raise DiscoveryError("invalid predictor configuration")
        self.config = dict(width=width, features=features, heads=heads,
                           value_width=value_width, stages=stages, kind=kind, topk=topk, rank=rank)
        if position_mode != "global" or read_groups != 1:
            # Recorded only when used, so saved original configurations keep their exact form.
            self.config.update(position_mode=position_mode, read_groups=read_groups)
        self.spans = nn.ModuleList([SpanPredictor(width, features, heads, value_width,
                                  kind=kind, topk=topk, rank=rank, position_mode=position_mode,
                                  read_groups=read_groups) for _ in range(stages)])

    def forecast(self, frontier):
        """Only the first boundary is supplied; later states are entirely predicted."""
        rows, x = [], frontier
        for span in self.spans:
            row = span(x)
            rows.append(row)
            x = row["state"]
        return rows

    def initialize_scales(self, samples):
        def rms(v):
            return v.square().mean().sqrt().clamp_min(1e-4)

        with torch.no_grad():
            for i, span in enumerate(self.spans):
                if span.position_mode == "global":
                    x = torch.cat([r["states"][i] for r in samples])
                    delta = torch.cat([r["states"][i + 1] - r["states"][i] for r in samples])
                    value = torch.cat([r["values"][i] for r in samples])
                    span.center.copy_(x.mean(0))
                    span.scale.copy_(rms(x - x.mean(0)))
                    span.write_scale.copy_(rms(delta))
                    span.value_center.copy_(value.mean(0))
                    span.value_scale.copy_(rms(value - value.mean(0)))
                    span.local_write.bias.copy_(delta.mean(0) / span.write_scale)
                    continue
                # Split mode: statistics for position 0 and for positions 1.. are kept apart.
                x0 = torch.stack([r["states"][i][0] for r in samples])
                xr = torch.cat([r["states"][i][1:] for r in samples])
                d0 = torch.stack([r["states"][i + 1][0] - r["states"][i][0] for r in samples])
                dr = torch.cat([r["states"][i + 1][1:] - r["states"][i][1:] for r in samples])
                value = torch.cat([r["values"][i][1:] for r in samples])
                span.center.copy_(torch.stack([x0.mean(0), xr.mean(0)]))
                span.scale.copy_(torch.stack([rms(x0 - x0.mean(0)), rms(xr - xr.mean(0))]))
                span.write_scale.copy_(torch.stack([rms(d0), rms(dr)]))
                span.value_center.copy_(value.mean(0))
                span.value_scale.copy_(rms(value - value.mean(0)))
                span.local_write.bias.copy_(dr.mean(0) / span.write_scale[1])
                span.first_shift.copy_(d0.mean(0) - dr.mean(0) * span.write_scale[0] / span.write_scale[1])


def tensor_sample(arrays, device):
    return {k: torch.as_tensor(np.array(arrays[k], copy=True), device=device, dtype=torch.float32)
            for k in ("states", "routes", "values")}


def sample_loss(model, sample, *, auxiliary=True, own_weight=0, output_consumer=None):
    losses, observations = [], []
    for i, span in enumerate(model.spans):
        row = span(sample["states"][i])
        target = sample["states"][i + 1]
        state = ((row["state"] - target) / span.row_write_scale(len(target), target.device)).square().mean()
        truth = sample["routes"][i]
        route = (truth * (truth.clamp_min(1e-8).log()
                         - row["routes"].clamp_min(1e-8).log())).sum(-1).mean()
        value = ((row["values"] - sample["values"][i]) / span.value_scale).square().mean()
        sparsity = row["code"].abs().mean()
        loss = state + (0.25 * route + 0.25 * value if auxiliary else 0) + 0.0001 * sparsity
        losses.append(loss)
        observations.append({"span": i, "state_loss": float(state.detach()),
                             "route_kl": float(route.detach()), "value_loss": float(value.detach()),
                             "code_l1": float(sparsity.detach())})
    loss = torch.stack(losses).mean()
    if own_weight or output_consumer is not None:
        predicted = model.forecast(sample["states"][0])
        own = own_state_loss(model, predicted, sample)
        loss = loss + own_weight * own
        observations.append({"own_state_loss": float(own.detach()), "weight": own_weight})
        if output_consumer is not None:
            output = output_consumer(predicted[-1]["state"], sample["states"][-1])
            loss = loss + 0.25 * output
            observations.append({"native_output_kl": float(output.detach()), "weight": 0.25})
    return loss, observations


def own_state_loss(model, predicted, sample):
    """Own-depth state MSE, each row normalised by its span's (per-class) write scale."""
    return torch.stack([((row["state"] - sample["states"][i + 1]) /
                         model.spans[i].row_write_scale(len(row["state"]), row["state"].device)).square().mean()
                        for i, row in enumerate(predicted)]).mean()


def centered_error_excluding_position_0(predicted, target):
    """Mean squared error on positions 1.. over the target's variance across positions 1.."""
    p, t = predicted[1:].double(), target[1:].double()
    variation = (t - t.mean(0, keepdim=True)).square().mean()
    return float((p - t).square().mean() / variation.clamp_min(1e-20))


def evaluate(model, samples, *, output_consumer=None):
    local, rollout, outputs, robust, details = [], [], [], [], []
    with torch.inference_mode():
        for sample in samples:
            loss, obs = sample_loss(model, sample, auxiliary=True)
            predictions = model.forecast(sample["states"][0])
            own = own_state_loss(model, predictions, sample)
            spans = [centered_error_excluding_position_0(p["state"], sample["states"][i + 1])
                     for i, p in enumerate(predictions)]
            local.append(float(loss))
            rollout.append(float(own))
            robust.append(float(np.mean(spans)))
            details.append({"local": obs, "own_state_normalized_mse": float(own),
                            "centered_relative_mse_excluding_position_0": spans})
            if output_consumer is not None:
                kl = float(output_consumer(predictions[-1]["state"], sample["states"][-1]))
                outputs.append(kl)
                details[-1]["native_output_kl"] = kl
    return {"local_loss": float(np.mean(local)), "own_state_loss": float(np.mean(rollout)),
            "centered_error_excluding_position_0": float(np.mean(robust)),
            "native_output_kl": float(np.mean(outputs)) if outputs else None,
            "documents": details}


def save_snapshot(path, snapshot):
    """Pack the public controller's full closure, without pickle or executable objects."""
    tensors = {}

    def pack(value):
        if isinstance(value, torch.Tensor):
            name = "tensor_" + str(len(tensors))
            tensors[name] = value.detach().cpu().numpy().copy()
            return {"type": "tensor", "key": name}
        if isinstance(value, np.ndarray):
            name = "array_" + str(len(tensors))
            tensors[name] = value.copy()
            return {"type": "ndarray", "key": name}
        if isinstance(value, np.generic):
            return pack(value.item())
        if isinstance(value, Mapping):
            return {"type": "dict", "items": [[pack(k), pack(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"type": "tuple" if isinstance(value, tuple) else "list",
                    "items": [pack(v) for v in value]}
        if value is None or type(value) in (str, bool, int, float):
            return {"type": "scalar", "value": value}
        raise DiscoveryError("unsupported training closure value: " + str(type(value)))

    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    tree = pack(snapshot.to_state_dict())
    arrays = save_arrays(path / "state.npz", **tensors)
    save_json(path / "state.json", sealed({"schema": "mdb-native-discovery-v1/training-state",
              "tree": tree, "arrays": arrays, "controller_fingerprint": snapshot.fingerprint}))


def load_snapshot(path):
    from saturn_pub.training._checkpoint import TrainingStateSnapshot
    path = Path(path)
    record = json.loads((path / "state.json").read_text())
    checked(record)
    arrays = read_arrays(path, record["arrays"])

    def unpack(row):
        kind = row["type"]
        if kind == "tensor":
            return torch.from_numpy(arrays[row["key"]].copy())
        if kind == "ndarray":
            return arrays[row["key"]].copy()
        if kind == "scalar":
            return row["value"]
        if kind == "dict":
            return {unpack(k): unpack(v) for k, v in row["items"]}
        if kind in ("tuple", "list"):
            values = [unpack(v) for v in row["items"]]
            return tuple(values) if kind == "tuple" else values
        raise DiscoveryError("unknown training closure tag")

    snapshot = TrainingStateSnapshot.from_state_dict(unpack(record["tree"]))
    if snapshot.fingerprint != record["controller_fingerprint"]:
        raise DiscoveryError("rehydrated training closure identity differs")
    return snapshot


def fit_predictors(discovery, development, *, root, binding, boundaries, features=32,
                   steps=60, interval_steps=10, seed=0, device="cpu", kinds=KINDS,
                   budget=lambda: None, output_consumer=None, arms=None, kv_groups=None):
    """Input rows are (trace_record, arrays); heldout rows are refused before fitting.

    ``arms`` (preset name or list of kinds/dicts, see ``arm_spec``) overrides ``kinds``; the
    default fits the original three configurations under their kind names.
    """
    from saturn_pub.training import (
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
        TrainingTransaction,
    )
    if steps < 1 or interval_steps < 1 or features < 1:
        raise DiscoveryError("fitting budgets must be positive")
    if not discovery or not development:
        raise DiscoveryError("fitting needs separate discovery and development documents")
    for expected, rows in (("discovery", discovery), ("development", development)):
        for record, arrays in rows:
            checked(record)
            if record["split"] != expected or record["binding"] != binding:
                raise DiscoveryError("fitting row split or execution binding differs")
            if record["boundaries"] != list(boundaries):
                raise DiscoveryError("fitting boundary geometry differs")
    if ({r[0]["group"] for r in discovery} & {r[0]["group"] for r in development}
            or {r[0]["text_sha256"] for r in discovery} & {r[0]["text_sha256"] for r in development}):
        raise DiscoveryError("source overlap between fitting and feedback")
    specs = resolve_arms(list(kinds) if arms is None else arms, kv_groups=kv_groups)
    train = [tensor_sample(a, device) for _, a in discovery]
    feedback = [tensor_sample(a, device) for _, a in development]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    first = discovery[0][1]
    width, heads, value_width = first["states"].shape[-1], first["routes"].shape[1], first["values"].shape[-1]
    predictors, models = {}, {}
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        budget()
        torch.manual_seed(seed)
        net = StatePredictor(width=width, features=features, heads=heads,
                             value_width=value_width, stages=len(boundaries) - 1, kind=kind,
                             position_mode=spec["position_mode"], read_groups=spec["read_groups"]).to(device)
        net.initialize_scales(train)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0001)
        sampler = torch.Generator(device="cpu").manual_seed(seed + 1)
        cursor = {"updates": 0}
        attempted = {"updates": 0, "evaluations": 0, "output_consumer_comparisons": 0}
        branch_policy = {"rejections": 0}
        arm = root / name
        arm.mkdir()

        def restore(value):
            cursor.clear()
            cursor.update(value)

        rollout_key = ("centered_error_excluding_position_0" if spec["selection"] == "position_robust"
                       else "state_loss")
        objectives = [PromotionObjective("autonomous_rollout", rollout_key, "minimize"),
                      PromotionObjective("validation", "consumer_loss", "minimize")]
        if output_consumer is not None:
            objectives.insert(0, PromotionObjective("validation", "output_kl", "minimize"))

        def compare_output(predicted, target):
            attempted["output_consumer_comparisons"] += 1
            return output_consumer(predicted, target)

        consumer = compare_output if output_consumer is not None else None
        controller = TrainingTransaction(controller_id="native_discovery_" + name,
            model=net, optimizer=optimizer, cursor_capture=lambda: dict(cursor), cursor_restore=restore,
            torch_generators={"document_sampler": sampler}, max_interval_steps=interval_steps,
            policy=PromotionPolicy("lexicographic", tuple(objectives)),
            receipt_store=ImmutableDirectoryReceiptStore(arm / "receipts"))

        def evaluator():
            budget()
            result = evaluate(net, feedback, output_consumer=consumer)
            index = attempted["evaluations"]
            # Capture complete candidate closure before the controller may reject it.
            save_snapshot(arm / "candidates" / f"{index:04d}", controller.capture())
            save_json(arm / "candidates" / f"{index:04d}" / "metrics.json", result)
            attempted["evaluations"] += 1
            validation = {"consumer_loss": result["local_loss"]}
            if consumer is not None:
                validation["output_kl"] = result["native_output_kl"]
            return EvaluationScores(validation=validation,
                autonomous_rollout={rollout_key: result["own_state_loss"] if rollout_key == "state_loss"
                                    else result["centered_error_excluding_position_0"]},
                trace_endpoints={"consumer_loss": result["local_loss"]},
                evidence={"scope": "development native trace targets; no heldout access",
                          "metrics": result, "snapshot": f"candidates/{index:04d}"})

        def update(_):
            budget()
            # This is an explicit external branch-selection policy. Apply it
            # inside the transaction so optimizer state remains in the closure.
            learning_rate = 0.003 * 0.5 ** branch_policy["rejections"]
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            index = int(torch.randint(len(train), (), generator=sampler))
            optimizer.zero_grad(set_to_none=True)
            loss, observations = sample_loss(net, train[index], auxiliary=kind != "sparse_reconstruction",
                own_weight=0.5, output_consumer=consumer if kind != "sparse_reconstruction" else None)
            if not torch.isfinite(loss):
                raise DiscoveryError("nonfinite update")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            cursor["updates"] += 1
            attempted["updates"] += 1
            return {"loss": float(loss.detach()), "document_index": index,
                    "learning_rate": learning_rate, "consumers": observations}

        controller.establish_baseline(evaluator)
        decisions = []
        for start in range(0, steps, interval_steps):
            receipt = controller.run_interval(interval_id=f"update_{start:05d}",
                steps=min(interval_steps, steps - start), train_step=update, evaluator=evaluator)
            decisions.append(receipt.to_dict())
            if not receipt.to_dict()["comparison"]["promoted"]:
                branch_policy["rejections"] += 1
        final = evaluate(net, feedback, output_consumer=consumer)
        save_snapshot(arm / "accepted", controller.capture())
        weights = save_arrays(arm / "weights.npz", **{
            k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()})
        options = {}
        if spec != arm_spec(kind):
            options["arm_options"] = {k: spec[k] for k in ("kind", "position_mode", "selection", "read_groups")}
        predictors[name] = {"config": net.config, "weights": weights, "development": final, **options,
                            "decisions": decisions, "physical_work": attempted,
                            "accepted_updates": cursor["updates"],
                            "branch_policy": {"after_rejection": "halve candidate learning rate", **branch_policy},
                            "accepted_state": controller.accepted_state_fingerprint}
        models[name] = net.eval()
        print("DISCOVERY_FIT " + json.dumps({"arm": name, "development": final["own_state_loss"],
                                             "development_centered_error_excluding_position_0":
                                                 final["centered_error_excluding_position_0"],
                                             "development_output_kl": final["native_output_kl"],
                                             "accepted_updates": cursor["updates"],
                                             "updates": attempted["updates"]}), flush=True)
    metadata = sealed({"schema": "mdb-native-discovery-v1/predictor", "binding": binding,
        "boundaries": list(boundaries), "fitted_document_ids": [r[0]["document_id"] for r in discovery],
        "selection_document_ids": [r[0]["document_id"] for r in development],
        "arms": predictors, "seed": seed,
        "training_objective": "local span write MSE plus 0.5 own-depth MSE; consumer arms add 0.25 route KL, 0.25 value MSE and optional 0.25 native output KL; 0.0001 code L1",
        "selection_policy": "development native output KL when available, then own-depth and local consumer error; continuation policy, not a scientific verdict",
        **({"arm_option_policy": {
            "position_mode=split": "separate input centre/scale and write scale for position 0 and positions 1..; initial write = position-split mean write; state errors normalised per class",
            "selection=position_robust": "development native output KL, then own-depth centred relative MSE on positions 1.. (position 0 excluded), then local consumer loss",
            "read_groups>1": "predicted read keeps native KV groups separate (heads averaged only within their KV group)"}}
           if any("arm_options" in a for a in predictors.values()) else {})})
    save_json(root / "model.json", metadata)
    return metadata, models


def load_predictors(root, *, device="cpu"):
    root = Path(root)
    meta = json.loads((root / "model.json").read_text())
    checked(meta)
    models = {}
    for kind, arm in meta["arms"].items():
        arrays = read_arrays(root / kind, arm["weights"])
        model = StatePredictor(**arm["config"]).to(device)
        model.load_state_dict({k: torch.from_numpy(v.copy()).to(device) for k, v in arrays.items()}, strict=True)
        models[kind] = model.eval()
    return meta, models


def forecast_arrays(model, frontier):
    device = next(model.parameters()).device
    with torch.inference_mode():
        rows = model.forecast(torch.as_tensor(np.array(frontier, copy=True),
                                             device=device, dtype=torch.float32))
    return {key: np.stack([r[key].detach().cpu().numpy() for r in rows])
            for key in ("state", "routes", "values", "code")}


def score_forecast(prediction, target):
    rows = []
    for i in range(len(prediction["state"])):
        p, t = prediction["routes"][i], target["routes"][i]
        kl = float((t * (np.log(np.maximum(t, 1e-8)) - np.log(np.maximum(p, 1e-8)))).sum(-1).mean())
        rows.append({"span": i, "state": error_metrics(prediction["state"][i], target["states"][i + 1]),
                     "route_kl": kl,
                     "source_top1_agreement": float((p.argmax(-1) == t.argmax(-1)).mean()),
                     "source_top1": evaluate_route_forecast(p, t, name="learned route"),
                     "value": error_metrics(prediction["values"][i], target["values"][i])})
    return rows
