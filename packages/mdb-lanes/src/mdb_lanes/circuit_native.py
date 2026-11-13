"""Lazy, original-module inspection for the public resident Qwen decoder lane.

Hooks observe an isolated native step. The optional adjoint reuses original
layer execution and cache construction; it neither fits a lens nor trains.
"""

from contextlib import contextmanager

from saturn_pub.values import describe


def _decoder(session):
    from saturn_pub.adapters.decoder import DecoderAdapter

    adapter = session.adapter
    if (
        not isinstance(adapter, DecoderAdapter)
        or adapter.spec.model_type not in {"qwen2", "qwen3"}
        or adapter.execution.get("residency", "resident") != "resident"
    ):
        raise ValueError(
            "native circuit expansion currently supports resident public Qwen2/Qwen3 DecoderAdapter"
        )
    adapter.validate_execution()
    state = session.capture(retain=False).payload
    layer = state["layer"]
    if state["phase"] != "layer" or not 0 <= layer < adapter.layers:
        raise ValueError(
            "select hidden at a pending decoder layer before expanding native attention/MLP"
        )
    hidden = session.read("hidden")
    if hidden is None or hidden.shape[:2] != (1, 1):
        raise ValueError("native circuit expansion requires one pending token and batch one")
    return adapter, layer


@contextmanager
def _ports(layer, *, detach):
    values, handles = {}, []

    def keep(name, value):
        if isinstance(value, tuple):
            value = value[0]
        values[name] = value.detach().clone() if detach else value

    def pre(name):
        def hook(module, args):
            keep(name, args[0])

        return hook

    def post(name):
        def hook(module, args, output):
            keep(name, output)

        return hook

    handles.append(layer.mlp.register_forward_pre_hook(pre("normalized")))
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(pre("post_attention")))
    for name, module in (
        ("gate", layer.mlp.gate_proj),
        ("up", layer.mlp.up_proj),
        ("activated_gate", layer.mlp.act_fn),
        ("mlp_write", layer.mlp.down_proj),
        ("attention_write", layer.self_attn.o_proj),
        ("layer_output", layer),
    ):
        handles.append(module.register_forward_hook(post(name)))
    handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre("product")))
    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre("head_concat")))

    def attention(module, args, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            keep("attention", output[1])

    handles.append(layer.self_attn.register_forward_hook(attention))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def _parameters(adapter, layer):
    prefix = f"{adapter.spec.backbone}.{adapter.spec.layers}.{layer}"
    return [
        {
            "model_identity": adapter.model_identity,
            "tensor": f"{prefix}.{name}",
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "storage": "original-model-view",
        }
        for name, param in adapter._layer_modules[layer].named_parameters()
    ]


def parameter_view(session, name, selector):
    """A bounded transient view, never a stored per-investigation weight bank."""
    from saturn_pub.adapters.decoder import DecoderAdapter

    from .circuit_investigation import _selector, sample

    adapter = session.adapter
    if not isinstance(adapter, DecoderAdapter):
        raise ValueError("parameter view requires a public decoder adapter")
    adapter.validate_execution()
    if not name:
        raise ValueError("parameter view requires an original named tensor")
    try:
        tensor = adapter.model.get_parameter(name)
        indices = tuple(_selector(selector))
        selected = tensor[indices] if indices else tensor
    except (AttributeError, IndexError) as error:
        raise ValueError("parameter or selector is outside the original checkpoint") from error
    return {
        "model_identity": adapter.model_identity,
        "tensor": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "selector": list(indices),
        "sample": sample(selected),
        "persisted": False,
        "authority": "original-model-view",
    }


def consumer_gradient(session, consumer):
    """First-readout adjoint from the exact selected residual and fixed past KV.

    Uses the original adapter's native layer executor under autograd instead of
    Session snapshot cloning (which correctly detaches durable native state).
    The caller must compare the primal with its native first-readout lane.
    """
    import torch

    adapter, index = _decoder(session)
    consumer = dict(consumer)
    if consumer.get("selector"):
        selector = consumer["selector"]
        if len(selector) != 2 or selector[0] != 0:
            raise ValueError("native adjoint requires a batch-one logit coordinate")
        consumer["token"] = selector[1]
    if consumer["port"] != "logits" or "token" not in consumer:
        raise ValueError("native adjoint requires an explicit first-readout token or contrast")
    if max(consumer["token"], consumer.get("contrast", 0)) >= adapter.model.config.vocab_size:
        raise ValueError("consumer token is outside native vocabulary")
    with torch.inference_mode(False), torch.enable_grad():
        hidden = session.read("hidden").detach().clone().requires_grad_(True)
        state = session.capture(retain=False).payload
        keys = [x.detach().clone() for x in state["keys"]]
        values = [x.detach().clone() for x in state["values"]]
        cache = adapter._cache(keys, values)
        position = session.read("tokens").shape[-1] - 1
        with _ports(adapter._layer_modules[index], detach=False) as ports:
            out = hidden
            for layer in range(index, adapter.layers):
                out = adapter._run_layer(layer, out, cache, position)
            logits = adapter._apply_logit_softcap(
                adapter.model.lm_head(getattr(adapter.backbone, adapter.spec.final_norm)(out))
            )[:, -1]
            target = logits[0, consumer["token"]]
            if "contrast" in consumer:
                target = target - logits[0, consumer["contrast"]]
            names = ["layer_output", "attention_write", "product", "gate", "up", "head_concat"]
            tensors = [hidden, *[ports[name] for name in names]]
            gradients = torch.autograd.grad(target, tensors)
            grads = dict(zip(["hidden", *names], gradients))
            result = {
                "value": float(target.detach()),
                "logits": describe(logits),
                "scope": "first-readout native-operator autograd with fixed historical KV; no argmax/rounding derivative",
                "hidden_gradient": describe(grads["hidden"]),
                "gradient_norm": float(grads["hidden"].double().norm()),
                "neuron_product_partials": grads["product"].detach().reshape(-1),
                "neuron_activity_projection": (
                    grads["product"].detach().double() * ports["product"].detach().double()
                ).reshape(-1),
            }
            gradient = grads["hidden"].detach().clone()
    adapter.validate_execution()
    return result, gradient


@contextmanager
def observe_layer(session):
    """Capture the actual first native transition of a supported panel arm."""
    adapter, index = _decoder(session)
    with _ports(adapter._layer_modules[index], detach=True) as ports:
        yield ports


def finish_observation(session, ports):
    index = session.capture(retain=False).payload["layer"] - 1
    ports["cache_values"] = session.capture(retain=False).payload["values"][index]


def finite_accounting(base, changed):
    """All finite terms and the changed native arithmetic residual, not a fit."""
    import torch

    out = {
        "scope": "first native pending-token layer; final consumer measured separately",
        "base_ports": {k: describe(v) for k, v in base.items()},
        "changed_ports": {k: describe(v) for k, v in changed.items()},
    }
    s0, s1 = base["activated_gate"].double(), changed["activated_gate"].double()
    u0, u1 = base["up"].double(), changed["up"].double()
    f0, f1 = base["product"].double(), changed["product"].double()
    pieces = {
        "gate": (s1 - s0) * u0,
        "up": s0 * (u1 - u0),
        "interaction": (s1 - s0) * (u1 - u0),
        "native_remainder_change": (f1 - s1 * u1) - (f0 - s0 * u0),
    }
    out["mlp"] = {
        "term_norms": {k: float(v.norm()) for k, v in pieces.items()},
        "closure_max_abs": float(((f1 - f0) - sum(pieces.values())).abs().max()),
        "native_delta_norm": float((f1 - f0).norm()),
    }
    if "attention" in base and "attention" in changed:
        p0, p1 = base["attention"].double()[0, :, 0], changed["attention"].double()[0, :, 0]
        v0, v1 = base["cache_values"].double()[0], changed["cache_values"].double()[0]
        groups = p0.shape[0] // v0.shape[0]
        v0, v1 = [v.repeat_interleave(groups, dim=0) for v in (v0, v1)]
        read = lambda p, v: torch.einsum("hs,hsd->hd", p, v)
        a0, a1 = [p["head_concat"].double().reshape(p0.shape[0], -1) for p in (base, changed)]
        terms = {
            "routing": read(p1 - p0, v0),
            "payload": read(p0, v1 - v0),
            "interaction": read(p1 - p0, v1 - v0),
            "native_remainder_change": (a1 - read(p1, v1)) - (a0 - read(p0, v0)),
        }
        out["attention"] = {
            "term_norms": {k: float(v.norm()) for k, v in terms.items()},
            "closure_max_abs": float(((a1 - a0) - sum(terms.values())).abs().max()),
            "native_delta_norm": float((a1 - a0).norm()),
        }
    return out


def inspect_decoder(session, consumer, *, view="mlp"):
    import torch

    from .circuit_investigation import measure, sample

    adapter, index = _decoder(session)
    parent = session.capture(retain=False)
    branch = session.fork(parent)
    module = adapter._layer_modules[index]
    with _ports(module, detach=True) as ports:
        receipt = branch.continue_(1)
    required = {
        "normalized",
        "gate",
        "up",
        "activated_gate",
        "product",
        "mlp_write",
        "head_concat",
        "attention_write",
        "layer_output",
    }
    if not required <= ports.keys():
        raise ValueError("native execution did not expose all declared inspection ports")
    result = {
        "layer": index,
        "execution_point": dict(session.inspect().execution_point),
        "native_receipt": receipt.to_dict(),
        "input": describe(session.read("hidden")),
        "parameters": _parameters(adapter, index),
        "ports": {k: describe(v) for k, v in ports.items()},
        "scope": "one native pending-token layer; fixed original past KV",
        "native_state_mutated": False,
    }
    g, u, s, product = [
        ports[name].double().reshape(-1) for name in ("gate", "up", "activated_gate", "product")
    ]
    top = product.abs().argsort(descending=True)[:8].tolist()
    result["mlp"] = {
        "read_interface": "original gate/up rows at normalized input",
        "neurons": [
            {
                "neuron": i,
                "gate": float(g[i]),
                "up": float(u[i]),
                "activated_gate": float(s[i]),
                "product": float(product[i]),
            }
            for i in top
        ],
        "native_product_remainder_norm": float((product - s * u).norm()),
        "ranking": "absolute native product, not semantic importance",
    }
    if "attention" in ports:
        p = ports["attention"].double()[0, :, 0]
        values = branch.capture(retain=False).payload["values"][index].double()[0]
        groups = p.shape[0] // values.shape[0]
        all_values = values.repeat_interleave(groups, dim=0)
        reconstructed = torch.einsum("hs,hsd->hd", p, all_values)
        native_return = ports["head_concat"].double().reshape(p.shape[0], -1)
        result["attention"] = {
            "probability_authority": "captured native eager attention",
            "heads": [
                {
                    "head": h,
                    "kv_group": h // groups,
                    "top_sources": [
                        {
                            "row": j,
                            "token_id": int(session.read("tokens")[0, j]),
                            "mass": float(p[h, j]),
                        }
                        for j in p[h].argsort(descending=True)[:8].tolist()
                    ],
                }
                for h in range(p.shape[0])
            ],
            "weighted_value_remainder_norm": float((native_return - reconstructed).norm()),
        }
        if view == "frequency":
            n = p.shape[-1]
            pairs = (
                torch.fft.fft(p, dim=-1).conj()[..., None] * torch.fft.fft(all_values, dim=1) / n
            )
            closed = pairs.sum(dim=1).real
            heads = []
            for h in range(min(8, p.shape[0])):
                chosen = pairs[h].abs().square().sum(-1).argsort(descending=True)[:8].tolist()
                remainder = pairs[h].sum(0) - pairs[h, chosen].sum(0)
                heads.append(
                    {
                        "head": h,
                        "bins": [
                            {
                                "bin": k,
                                "return_norm": float(pairs[h, k].norm()),
                                "complex_sample": [
                                    [float(x.real), float(x.imag)] for x in pairs[h, k, :4]
                                ],
                            }
                            for k in chosen
                        ],
                        "unlisted_sum": describe(remainder),
                        "unlisted_sum_sample": [
                            [float(x.real), float(x.imag)] for x in remainder[:4]
                        ],
                    }
                )
            result["frequency"] = {
                "window": n,
                "axis": "visible source positions, one fixed receiver",
                "convention": "unnormalized DFT; conjugate attention spectrum times value spectrum / N",
                "return_closure_max_abs": float((closed - reconstructed).abs().max()),
                "heads": heads,
                "displayed_heads": min(8, p.shape[0]),
                "total_heads": p.shape[0],
                "mlp_token_spectrum": "not-applicable: single pending-token MLP occurrence",
                "interpretation": "source correlation accounting, not square-window frequency transport or semantic bands",
            }
    else:
        result["attention"] = {
            "status": "evidence-incomplete",
            "reason": "native attention weights unavailable",
        }
    if view == "effect":
        # Continue the private native branch to its first original readout.
        for _ in range(adapter.layers + 5):
            if branch.capture(retain=False).payload["logits"] is not None:
                break
            branch.continue_(1)
        baseline = measure(branch, consumer)
        details, gradient = consumer_gradient(session, baseline["consumer"])
        partials = details.pop("neuron_product_partials")
        projections = details.pop("neuron_activity_projection")
        ranked = projections.abs().argsort(descending=True)[:8].tolist()
        result["effect"] = {
            **details,
            "consumer": baseline["consumer"],
            "native_readout": baseline,
            "primal_logits_exact": baseline["readout"] == details["logits"],
            "gradient_sample": sample(gradient),
            "neurons": [
                {
                    "neuron": i,
                    "product_partial": float(partials[i]),
                    "local_activity_projection": float(projections[i]),
                }
                for i in ranked
            ],
            "projection_meaning": "gradient times activity; not additive decomposition of finite output",
        }
    session.capture(retain=False)
    return result
