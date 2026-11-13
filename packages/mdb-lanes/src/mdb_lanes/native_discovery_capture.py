"""Passive Qwen-shaped decoder capture. No semantic addresses or state replacements."""
from __future__ import annotations

from .native_discovery import DiscoveryError, digest, save_arrays, save_json, sealed


def boundaries_for(layers, stages):
    if type(stages) is not int or not 1 <= stages <= layers:
        raise DiscoveryError("stages must be between one and the native layer count")
    return tuple(round(i * layers / stages) for i in range(stages + 1))


def capture(model, input_ids, *, boundaries, neuron_budget=32):
    """Capture a whole ordinary forward. Hooks return None and never replace outputs.

    states are residuals before boundary layers and before the final norm.
    routes are actual first-layer attention patterns for each predicted span.
    neurons are a uniformly selected native MLP pre-down-projection basis,
    available only after paid native prefix execution (a local diagnostic).
    """
    import numpy as np
    import torch

    if any(m.training for m in model.modules()):
        raise DiscoveryError("capture requires an evaluation-mode model")
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None or not callable(getattr(base, "forward", None)):
        raise DiscoveryError("capture currently supports decoder.model.layers")
    if (tuple(sorted(set(boundaries))) != tuple(boundaries)
            or boundaries[0] != 0 or boundaries[-1] != len(layers)):
        raise DiscoveryError("boundaries must uniquely cover layer zero to the final layer")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise DiscoveryError("capture uses one unpadded document with at least two tokens")
    if neuron_budget < 1:
        raise DiscoveryError("neuron budget must be positive")
    states, neurons, values, handles = {}, {}, {}, []

    def pre(index):
        def hook(module, args):
            states[index] = args[0][0].detach().float().cpu().numpy().copy()
        return hook

    def final(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        states[len(layers)] = value[0].detach().float().cpu().numpy().copy()

    def neuron(index):
        def hook(module, args):
            value = args[0][0]
            ids = torch.linspace(0, value.shape[-1] - 1,
                                 min(neuron_budget, value.shape[-1]), device=value.device).long()
            neurons[index] = value[:, ids].detach().float().cpu().numpy().copy()
        return hook

    def value_projection(index):
        def hook(module, args, output):
            values[index] = output[0].detach().float().cpu().numpy().copy()
        return hook

    try:
        for index in boundaries[:-1]:
            handles.append(layers[index].register_forward_pre_hook(pre(index)))
            handles.append(layers[index].mlp.down_proj.register_forward_pre_hook(neuron(index)))
            handles.append(layers[index].self_attn.v_proj.register_forward_hook(value_projection(index)))
        handles.append(layers[-1].register_forward_hook(final))
        with torch.inference_mode():
            output = base(input_ids=input_ids, use_cache=False, output_attentions=True,
                          return_dict=True)
        attention = getattr(output, "attentions", None)
        if attention is None or len(attention) != len(layers) or any(
                attention[i] is None for i in boundaries[:-1]):
            raise DiscoveryError("native attention capture unavailable; select eager attention")
        arrays = {
            "states": np.stack([states[i] for i in boundaries]),
            "routes": np.stack([attention[i][0].detach().float().cpu().numpy()
                                for i in boundaries[:-1]]),
            "neurons": np.stack([neurons[i] for i in boundaries[:-1]]),
            "values": np.stack([values[i] for i in boundaries[:-1]]),
            "tokens": input_ids[0].detach().cpu().numpy().astype(np.int64),
        }
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                raise DiscoveryError("nonfinite native capture: " + name)
        route = arrays["routes"]
        if (np.any(route < 0) or not np.allclose(route.sum(-1), 1, atol=0.01)
                or np.any(np.triu(route, 1) != 0)):
            raise DiscoveryError("captured routes do not satisfy causal attention mechanics")
        return arrays
    finally:
        for handle in handles:
            handle.remove()


def capture_document(model, tokenizer, row, *, root, binding, boundaries, max_tokens,
                     neuron_budget=32):
    import numpy as np
    import torch

    encoded = tokenizer(row["text"], add_special_tokens=False, truncation=True,
                        max_length=max_tokens, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) < 2:
        raise DiscoveryError("document is too short after tokenization")
    tensor = torch.tensor([ids], device=next(model.parameters()).device)
    arrays = capture(model, tensor, boundaries=boundaries, neuron_budget=neuron_budget)
    arrays["offsets"] = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    identifier = digest([row["id"], row["text_sha256"], binding, boundaries])
    descriptor = save_arrays(root / (identifier + ".npz"), **arrays)
    record = sealed({"schema": "mdb-native-discovery-v1/trace", "document_id": row["id"],
                     "group": row["group"], "split": row["split"],
                     "text_sha256": row["text_sha256"], "binding": binding,
                     "boundaries": list(boundaries), "arrays": descriptor,
                     "token_pieces": tokenizer.convert_ids_to_tokens(ids),
                     "authority": "passive native observation; not causal necessity",
                     "work": {"physical_forwards": 1, "physical_batch": 1,
                              "tokens": len(ids), "generation_tokens": 0,
                              "captured_bytes": sum(a.nbytes for a in arrays.values())}})
    save_json(root / (identifier + ".json"), record)
    return record, arrays
