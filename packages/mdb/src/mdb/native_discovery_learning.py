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


class SpanPredictor(nn.Module):
    def __init__(self, width, features, heads, value_width, *, kind, topk=8, rank=4):
        super().__init__()
        self.kind, self.heads, self.rank = kind, heads, rank
        self.topk = min(topk, features)
        self.encoder = nn.Linear(width, features)
        self.query = nn.Linear(features, heads * rank, bias=False)
        self.key = nn.Linear(features, heads * rank, bias=False)
        self.value = nn.Linear(features, value_width)
        self.local_write = nn.Linear(features, width)
        self.read_write = nn.Linear(value_width, width, bias=False)
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("write_scale", torch.ones(()))
        self.register_buffer("value_scale", torch.ones(()))
        self.register_buffer("value_center", torch.zeros(value_width))
        nn.init.zeros_(self.local_write.weight)
        nn.init.zeros_(self.local_write.bias)
        nn.init.zeros_(self.read_write.weight)

    def encode(self, x):
        raw = self.encoder((x - self.center) / self.scale)
        if self.kind == "linear_features":
            return raw
        positive = raw.relu()
        ids = positive.topk(self.topk, dim=-1).indices
        return torch.zeros_like(positive).scatter(-1, ids, positive.gather(-1, ids))

    def forward(self, x):
        code = self.encode(x)
        t = x.shape[0]
        q = self.query(code).reshape(t, self.heads, self.rank).transpose(0, 1)
        k = self.key(code).reshape(t, self.heads, self.rank).transpose(0, 1)
        score = q @ k.transpose(-1, -2) / self.rank ** 0.5
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        route = score.masked_fill(mask, -torch.inf).softmax(-1)
        value = self.value(code) * self.value_scale + self.value_center
        read = route.mean(0) @ ((value - self.value_center) / self.value_scale)
        write = (self.local_write(code) + self.read_write(read)) * self.write_scale
        return {"state": x + write, "routes": route, "values": value, "code": code}


class StatePredictor(nn.Module):
    def __init__(self, *, width, features, heads, value_width, stages, kind, topk=8, rank=4):
        super().__init__()
        if kind not in KINDS or min(width, features, heads, value_width, stages, topk, rank) < 1:
            raise DiscoveryError("invalid predictor configuration")
        self.config = dict(width=width, features=features, heads=heads,
                           value_width=value_width, stages=stages, kind=kind, topk=topk, rank=rank)
        self.spans = nn.ModuleList([SpanPredictor(width, features, heads, value_width,
                                  kind=kind, topk=topk, rank=rank) for _ in range(stages)])

    def forecast(self, frontier):
        """Only the first boundary is supplied; later states are entirely predicted."""
        rows, x = [], frontier
        for span in self.spans:
            row = span(x)
            rows.append(row)
            x = row["state"]
        return rows

    def initialize_scales(self, samples):
        with torch.no_grad():
            for i, span in enumerate(self.spans):
                x = torch.cat([r["states"][i] for r in samples])
                delta = torch.cat([r["states"][i + 1] - r["states"][i] for r in samples])
                value = torch.cat([r["values"][i] for r in samples])
                span.center.copy_(x.mean(0))
                span.scale.copy_((x - x.mean(0)).square().mean().sqrt().clamp_min(1e-4))
                span.write_scale.copy_(delta.square().mean().sqrt().clamp_min(1e-4))
                span.value_center.copy_(value.mean(0))
                span.value_scale.copy_((value - value.mean(0)).square().mean().sqrt().clamp_min(1e-4))
                span.local_write.bias.copy_(delta.mean(0) / span.write_scale)


def tensor_sample(arrays, device):
    return {k: torch.as_tensor(np.array(arrays[k], copy=True), device=device, dtype=torch.float32)
            for k in ("states", "routes", "values")}


def sample_loss(model, sample, *, auxiliary=True, own_weight=0, output_consumer=None):
    losses, observations = [], []
    for i, span in enumerate(model.spans):
        row = span(sample["states"][i])
        target = sample["states"][i + 1]
        state = ((row["state"] - target) / span.write_scale).square().mean()
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
        own = torch.stack([((row["state"] - sample["states"][i + 1]) /
                           model.spans[i].write_scale).square().mean()
                           for i, row in enumerate(predicted)]).mean()
        loss = loss + own_weight * own
        observations.append({"own_state_loss": float(own.detach()), "weight": own_weight})
        if output_consumer is not None:
            output = output_consumer(predicted[-1]["state"], sample["states"][-1])
            loss = loss + 0.25 * output
            observations.append({"native_output_kl": float(output.detach()), "weight": 0.25})
    return loss, observations


def evaluate(model, samples, *, output_consumer=None):
    local, rollout, outputs, details = [], [], [], []
    with torch.inference_mode():
        for sample in samples:
            loss, obs = sample_loss(model, sample, auxiliary=True)
            predictions = model.forecast(sample["states"][0])
            own = torch.stack([((p["state"] - sample["states"][i + 1]) /
                               model.spans[i].write_scale).square().mean()
                               for i, p in enumerate(predictions)]).mean()
            local.append(float(loss))
            rollout.append(float(own))
            details.append({"local": obs, "own_state_normalized_mse": float(own)})
            if output_consumer is not None:
                kl = float(output_consumer(predictions[-1]["state"], sample["states"][-1]))
                outputs.append(kl)
                details[-1]["native_output_kl"] = kl
    return {"local_loss": float(np.mean(local)), "own_state_loss": float(np.mean(rollout)),
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
                   budget=lambda: None, output_consumer=None):
    """Input rows are (trace_record, arrays); heldout rows are refused before fitting."""
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
    train = [tensor_sample(a, device) for _, a in discovery]
    feedback = [tensor_sample(a, device) for _, a in development]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    first = discovery[0][1]
    width, heads, value_width = first["states"].shape[-1], first["routes"].shape[1], first["values"].shape[-1]
    predictors, models = {}, {}
    for kind in kinds:
        budget()
        torch.manual_seed(seed)
        net = StatePredictor(width=width, features=features, heads=heads,
                             value_width=value_width, stages=len(boundaries) - 1, kind=kind).to(device)
        net.initialize_scales(train)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0001)
        sampler = torch.Generator(device="cpu").manual_seed(seed + 1)
        cursor = {"updates": 0}
        attempted = {"updates": 0, "evaluations": 0, "output_consumer_comparisons": 0}
        branch_policy = {"rejections": 0}
        arm = root / kind
        arm.mkdir()

        def restore(value):
            cursor.clear()
            cursor.update(value)

        objectives = [PromotionObjective("autonomous_rollout", "state_loss", "minimize"),
                      PromotionObjective("validation", "consumer_loss", "minimize")]
        if output_consumer is not None:
            objectives.insert(0, PromotionObjective("validation", "output_kl", "minimize"))

        def compare_output(predicted, target):
            attempted["output_consumer_comparisons"] += 1
            return output_consumer(predicted, target)

        consumer = compare_output if output_consumer is not None else None
        controller = TrainingTransaction(controller_id="native_discovery_" + kind,
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
                autonomous_rollout={"state_loss": result["own_state_loss"]},
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
        predictors[kind] = {"config": net.config, "weights": weights, "development": final,
                            "decisions": decisions, "physical_work": attempted,
                            "accepted_updates": cursor["updates"],
                            "branch_policy": {"after_rejection": "halve candidate learning rate", **branch_policy},
                            "accepted_state": controller.accepted_state_fingerprint}
        models[kind] = net.eval()
        print("DISCOVERY_FIT " + json.dumps({"arm": kind, "development": final["own_state_loss"],
                                             "updates": attempted["updates"]}), flush=True)
    metadata = sealed({"schema": "mdb-native-discovery-v1/predictor", "binding": binding,
        "boundaries": list(boundaries), "fitted_document_ids": [r[0]["document_id"] for r in discovery],
        "selection_document_ids": [r[0]["document_id"] for r in development],
        "arms": predictors, "seed": seed,
        "training_objective": "local span write MSE plus 0.5 own-depth MSE; consumer arms add 0.25 route KL, 0.25 value MSE and optional 0.25 native output KL; 0.0001 code L1",
        "selection_policy": "development native output KL when available, then own-depth and local consumer error; continuation policy, not a scientific verdict"})
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
