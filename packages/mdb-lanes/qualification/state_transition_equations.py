"""Independent R3 equations for own-state futures and typed input-path contrasts.

No model imports or observed-future inputs. The host supplies explicit program
segments; only architectural registers/flags cross segment boundaries.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

raw = runpy.run_path(str(Path(__file__).with_name("raw_isa_equations.py")))
suffix = raw["_suffix"]
linear, softmax = raw["linear"], raw["softmax"]


def prefix(window, regs, flags, weights, *, z_override=None, value_window=None, direct_override=None):
    original = raw["raw_prefix"](window, regs, flags, weights)
    if z_override is None and value_window is None and direct_override is None:
        return original
    z = original["z"] if z_override is None else np.asarray(z_override, dtype=np.float64)
    values_window = np.asarray(window if value_window is None else value_window, dtype=np.float64)
    if z.shape != (1, 256) or values_window.shape != (1, 120):
        raise ValueError("controlled prefix geometry differs")
    if not np.isfinite(z).all() or not np.isin(values_window, (0., 1.)).all():
        raise ValueError("controlled prefix requires finite z and binary byte values")
    regs = np.asarray(regs, dtype=np.float64)
    pd, ps = (softmax(linear(z, weights, name)) for name in ("dest", "src"))
    a = np.einsum("br,bri->bi", pd, regs)
    s = np.einsum("br,bri->bi", ps, regs)
    position = weights["pos"][None, :, :]
    context = np.broadcast_to(z[:, None, :], (1, 64, 256))
    query = linear(np.concatenate((position, context), axis=-1), weights, "query").reshape(1, 64, 2, 16)
    values = np.concatenate((values_window, np.zeros((1, 1)), np.ones((1, 1))), axis=-1)
    attention = softmax(np.einsum("bihd,jd->bihj", query, weights["ins_pos"]) * 4.)
    immediate = np.einsum("bihj,bj->bih", attention, values)
    slots = np.stack([values_window[:, start:start + 64] for start in range(0, 64, 8)], axis=-1)
    offset_head = linear(z, weights, "offset_head")
    pick = np.einsum("bo,bio->bi", softmax(offset_head), slots)
    tokens = np.concatenate((regs.transpose(0, 2, 1), a[..., None], s[..., None], immediate,
                             position, context, slots * 2. - 1., pick[..., None] * 2. - 1.), axis=-1)
    if direct_override is not None:
        direct = np.asarray(direct_override, dtype=np.float64)
        if direct.shape != (1, 64, 11) or not np.isfinite(direct).all():
            raise ValueError("direct input payload must be finite [1,64,11]")
        tokens = np.concatenate((tokens[:, :, :10], direct[:, :, :2], tokens[:, :, 12:284], direct[:, :, 2:]), axis=-1)
    return {**original, "z": z, "tokens": tokens, "input": linear(tokens, weights, "inp"),
            "query": query.reshape(1, 64, 32), "offset_head": offset_head}


def predict(window, regs, flags, weights, *, zero_posts=(), z_override=None, direct_override=None,
            projection_override=None, carrier_override=None, post_override=None):
    incoming = prefix(window, regs, flags, weights, z_override=z_override, direct_override=direct_override)
    state, stages = incoming["input"], dict(incoming)
    for layer in range(3):
        mapped = dict(weights)
        for name in suffix["required_weight_names"]():
            if name.startswith("mixers.2."):
                mapped[name] = weights[name.replace("mixers.2.", f"mixers.{layer}.")]
            elif name.startswith("post.2."):
                mapped[name] = weights[name.replace("post.2.", f"post.{layer}.")]
        options = {}
        if layer in zero_posts:
            options["post_override"] = np.zeros_like(state, dtype=np.float32)
        if layer == 2:
            options.update(projection_override=projection_override, carrier_override=carrier_override)
            if post_override is not None:
                options["post_override"] = post_override
        result = suffix["predict_suffix"](state, incoming["z"], regs, flags, mapped, **options)
        stages.update({f"layer{layer}.{name}": value for name, value in result["stages"].items()})
        state = result["stages"]["h"]
    result["stages"] = stages
    result["decoded"] = raw["decode"](result["heads"])
    return result


def retained_prediction(value):
    """Retain every declared native comparison port; do not publish unused expansions."""
    selected = {"z", "tokens", "input", "query", "offset_head", "layer2.h"}
    for layer in range(3):
        selected.update(f"layer{layer}.{name}" for name in ("S", "M", "U", "P2", "in_projected", "selected",
            "dt_affine", "post2_norm", "post2_affine", "post2_gelu", "P2_computed"))
    return {**value, "stages": {name: value["stages"][name] for name in sorted(selected)}}


def forecast(plan, supplied_weights):
    weights = raw["prepare_weights"](supplied_weights)
    initial = raw["roots"](plan["segments"][0]["program_hex"], plan["register_words"], plan["flags"])
    native = predict(*initial, weights)
    payloads = {"zero": np.zeros((1, 64, 48), dtype=np.float32),
                "sequence.U_native": native["stages"]["layer2.U"].astype(np.float32),
                "sequence.P2_native": native["stages"]["layer2.P2"].astype(np.float32)}
    trajectories = {}
    for arm in plan["arms"]:
        words, flags = list(plan["register_words"]), list(plan["flags"])
        steps, global_step, stopped = {}, 0, False
        for segment in plan["segments"]:
            offset = 0
            for segment_step in range(segment["max_instructions"]):
                roots = raw["roots"](segment["program_hex"], words, flags, offset)
                options = {}
                if global_step == 0 and arm != "native":
                    options = {"zero_posts": (0,), "projection_override": payloads["sequence.U_native"]}
                    if arm == "U_swap_P2native":
                        options["post_override"] = payloads["sequence.P2_native"]
                predicted = predict(*roots, weights, **options)
                decoded = predicted["decoded"]
                steps[str(global_step)] = {"before": {"register_words": list(words), "flags": list(flags), "offset": offset},
                    "roots": dict(zip(("window", "regs", "flags"), roots, strict=True)), **predicted,
                    "segment_id": segment["id"], "segment_step": segment_step,
                    "host_segment_start": segment_step == 0, "program_hex": segment["program_hex"]}
                global_step += 1
                if decoded["store"] or decoded["syscall"]:
                    stopped = True
                    break
                words, flags = decoded["register_words"], decoded["flags"]
                offset += decoded["length"]
                if offset >= len(bytes.fromhex(segment["program_hex"])):
                    break
            if stopped:
                break
        trajectories["sequence." + arm] = {"steps": steps, "arm": arm}

    other = raw["roots"](plan["contexts"]["new"], plan["register_words"], plan["flags"])
    payloads["path.z_new"] = raw["raw_prefix"](*other, weights)["z"].astype(np.float32)
    path_rows = {"path.old.old": trajectories["sequence.U_swap"]["steps"]["0"]}
    for z_source, value_source in (("new", "old"), ("old", "new"), ("new", "new")):
        key = f"path.{z_source}.{value_source}"
        options = {"zero_posts": (0,), "projection_override": payloads["sequence.U_native"]}
        if z_source == "new":
            options["z_override"] = payloads["path.z_new"]
        if value_source == "new":
            donor = prefix(*initial, weights, z_override=options.get("z_override"), value_window=other[0])
            tokens = donor["tokens"]
            direct = np.concatenate((tokens[:, :, 10:12], tokens[:, :, 284:293]), axis=-1).astype(np.float32)
            payloads[key + ".direct"] = direct
            options["direct_override"] = direct
        path_rows[key] = {"roots": dict(zip(("window", "regs", "flags"), initial, strict=True)),
                          **predict(*initial, weights, **options)}
    payloads["path.M_other"] = path_rows["path.new.new"]["stages"]["layer2.M"].astype(np.float32)
    payloads["path.P2_own"] = path_rows["path.old.old"]["stages"]["layer2.P2"].astype(np.float32)
    for key, p2 in (("path.carrier_other", None), ("path.carrier_other_P2own", payloads["path.P2_own"])):
        path_rows[key] = {"roots": dict(zip(("window", "regs", "flags"), initial, strict=True)),
            **predict(*initial, weights, zero_posts=(0,), projection_override=payloads["sequence.U_native"],
                      carrier_override=payloads["path.M_other"], post_override=p2)}
    mlp_rows = {}
    for operation, program in plan["mlp_programs"].items():
        roots = raw["roots"](program, plan["register_words"], plan["flags"])
        for arm in ("native", "zero_post0", "zero_post1", "zero_post2"):
            options = {} if arm == "native" else {"zero_posts": (int(arm[-1]),)}
            mlp_rows[f"mlp.{operation}.{arm}"] = {"roots": dict(zip(("window", "regs", "flags"), roots, strict=True)),
                                               **predict(*roots, weights, **options)}
    for trajectory in trajectories.values():
        trajectory["steps"] = {key: retained_prediction(value) for key, value in trajectory["steps"].items()}
    path_rows = {key: retained_prediction(value) for key, value in path_rows.items()}
    mlp_rows = {key: retained_prediction(value) for key, value in mlp_rows.items()}
    return {"predictions": {"trajectories": trajectories, "path_rows": path_rows, "mlp_rows": mlp_rows},
            "payloads": payloads}
