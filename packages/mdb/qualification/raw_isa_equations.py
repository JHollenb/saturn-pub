"""Independent raw-input R3 equations and autonomous architectural forecasts.

No Torch/model/native consumer imports. Reuses the previously sealed NumPy suffix
equations at each same-geometry layer, supplying the corresponding frozen weights.
All 15 instruction-window bytes participate, including later instruction bytes.
"""
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

_suffix = runpy.run_path(str(Path(__file__).with_name("explicit_suffix_equations.py")))
CHECKPOINT = _suffix["CHECKPOINT"]
SOURCE_FORMULAS = _suffix["SOURCE_FORMULAS"]
linear, gelu, softmax = (_suffix[key] for key in ("linear", "gelu", "softmax"))
predict_suffix = _suffix["predict_suffix"]
ARMS = ("native", "U_swap", "U_swap_P2native")
PROGRAM_HEX = "4881ef010000004831f74881c701000000"


def prepare_weights(weights):
    required = set(_suffix["required_weight_names"]())
    for layer in (0, 1):
        required.update(name.replace("mixers.2.", f"mixers.{layer}.") for name in required.copy() if name.startswith("mixers.2."))
        required.update(name.replace("post.2.", f"post.{layer}.") for name in required.copy() if name.startswith("post.2."))
    required.add("ins_pos")
    for module in ("decoder.0", "decoder.2", "query", "inp", "offset_head"):
        required.update((module + ".weight", module + ".bias"))
    if set(weights) != required:
        raise ValueError("R3 full checkpoint dictionary differs from the declared configuration")
    result = {}
    for name, value in weights.items():
        array = np.asarray(value, dtype=np.float64).copy()
        if not np.isfinite(array).all():
            raise ValueError("nonfinite checkpoint parameter: " + name)
        array.flags.writeable = False
        result[name] = array
    return result


def roots(program_hex, words, flags, offset=0):
    if len(words) != 8 or any(type(word) is not int or not 0 <= word < 2**64 for word in words):
        raise ValueError("eight canonical u64 register words required")
    if len(flags) != 4 or any(type(bit) is not int or bit not in (0, 1) for bit in flags):
        raise ValueError("four raw flag bits required")
    program = bytes.fromhex(program_hex)
    if type(offset) is not int or offset < 0 or offset >= len(program):
        raise ValueError("fetch offset must be inside the byte program")
    window = program[offset:offset + 15].ljust(15, b"\0")
    return (np.array([[(byte >> bit) & 1 for byte in window for bit in range(8)]], dtype=np.float32),
            np.array([[[(word >> bit) & 1 for bit in range(64)] for word in words]], dtype=np.float32),
            np.array([flags], dtype=np.float32))


def decode(heads):
    regs = np.asarray(heads["regs"])
    flags = np.asarray(heads["flags"])
    if regs.shape != (1, 8, 64) or flags.shape != (1, 4):
        raise ValueError("single architectural state required")
    words = [sum(int(bit) << index for index, bit in enumerate(row > 0)) for row in regs[0]]
    return {"register_words": words, "flags": (flags[0] > 0).astype(int).tolist(),
            "length": int(np.asarray(heads["length"])[0].argmax()) + 1,
            "store": bool(np.asarray(heads["store"])[0] > 0),
            "syscall": bool(np.asarray(heads["syscall"])[0] > 0)}


def raw_prefix(window_bits, reg_bits, flags, weights):
    window = np.asarray(window_bits, dtype=np.float64)
    regs, flags = np.asarray(reg_bits, dtype=np.float64), np.asarray(flags, dtype=np.float64)
    if window.shape != (1, 120) or regs.shape != (1, 8, 64) or flags.shape != (1, 4):
        raise ValueError("raw R3 input ABI differs")
    if any(not np.isin(value, (0., 1.)).all() for value in (window, regs, flags)):
        raise ValueError("raw inputs must be binary")
    decoder_first = linear(window * 2. - 1., weights, "decoder.0")
    decoder_hidden = gelu(decoder_first)
    decoder_second = linear(decoder_hidden, weights, "decoder.2")
    z = gelu(decoder_second)
    dest, src = linear(z, weights, "dest"), linear(z, weights, "src")
    pd, ps = softmax(dest), softmax(src)
    a = np.einsum("br,bri->bi", pd, regs)
    s = np.einsum("br,bri->bi", ps, regs)
    position = weights["pos"][None, :, :]
    context = np.broadcast_to(z[:, None, :], (1, 64, 256))
    query = linear(np.concatenate((position, context), axis=-1), weights, "query").reshape(1, 64, 2, 16)
    values = np.concatenate((window, np.zeros((1, 1)), np.ones((1, 1))), axis=-1)
    attention = softmax(np.einsum("bihd,jd->bihj", query, weights["ins_pos"]) * 4.)
    immediate = np.einsum("bihj,bj->bih", attention, values)
    # Aligned slots are eight 64-bit slices at an eight-bit stride.
    slots = np.stack([window[:, start:start + 64] for start in range(0, 64, 8)], axis=-1)
    offset = softmax(linear(z, weights, "offset_head"))
    pick = np.einsum("bo,bio->bi", offset, slots)
    tokens = np.concatenate((regs.transpose(0, 2, 1), a[..., None], s[..., None], immediate,
                             position, context, slots * 2. - 1., pick[..., None] * 2. - 1.), axis=-1)
    embedded = linear(tokens, weights, "inp")
    return {"z": z, "tokens": tokens, "input": embedded,
            "decoder.0": decoder_first, "decoder.1": decoder_hidden,
            "decoder.2": decoder_second, "decoder.3": z,
            "query": query.reshape(1, 64, 32), "offset_head": linear(z, weights, "offset_head")}


def predict_raw(window_bits, reg_bits, flags, weights, *, zero_post0=False,
                projection_override=None, post_override=None):
    prefix = raw_prefix(window_bits, reg_bits, flags, weights)
    state = prefix["input"]
    stages = dict(prefix)
    for layer in (0, 1):
        mapped = dict(weights)
        for name in _suffix["required_weight_names"]():
            if name.startswith("mixers.2."):
                mapped[name] = weights[name.replace("mixers.2.", f"mixers.{layer}.")]
            elif name.startswith("post.2."):
                mapped[name] = weights[name.replace("post.2.", f"post.{layer}.")]
        result = predict_suffix(state, prefix["z"], reg_bits, flags, mapped,
                                post_override=np.zeros_like(state) if zero_post0 and layer == 0 else None)
        stages.update({f"layer{layer}.{name}": value for name, value in result["stages"].items()})
        state = result["stages"]["h"]
    final = predict_suffix(state, prefix["z"], reg_bits, flags, weights,
                           projection_override=projection_override, post_override=post_override)
    stages.update({f"layer2.{name}": value for name, value in final["stages"].items()})
    final["stages"] = stages
    final["decoded"] = decode(final["heads"])
    return final


def forecast_trajectory(program_hex, words, flags, weights, *, arm="native", instructions=3):
    if arm not in ARMS or type(instructions) is not int or not 1 <= instructions <= 16:
        raise ValueError("unsupported arm or instruction budget")
    offset, steps, payloads = 0, {}, {}
    for step in range(instructions):
        raw = roots(program_hex, words, flags, offset)
        options = {}
        if step == 0 and arm != "native":
            native = predict_raw(*raw, weights)
            payloads = {"U_native": native["stages"]["layer2.U"].astype(np.float32),
                        "P2_native": native["stages"]["layer2.P2"].astype(np.float32),
                        "post0_zero": np.zeros((1, 64, 48), dtype=np.float32)}
            options = {"zero_post0": True, "projection_override": payloads["U_native"]}
            if arm == "U_swap_P2native":
                options["post_override"] = payloads["P2_native"]
        predicted = predict_raw(*raw, weights, **options)
        decoded = predicted["decoded"]
        steps[str(step)] = {"before": {"register_words": list(words), "flags": list(flags), "offset": offset},
                            "roots": {"window": raw[0], "regs": raw[1], "flags": raw[2]}, **predicted}
        if decoded["store"] or decoded["syscall"]:
            steps[str(step)]["termination"] = "stop-before-effect-commit"
            break
        words, flags = decoded["register_words"], decoded["flags"]
        offset += decoded["length"]
        if offset >= len(bytes.fromhex(program_hex)):
            steps[str(step)]["termination"] = "program-end"
            break
    return {"steps": steps, "payloads": payloads, "arm": arm, "program_hex": program_hex}


def self_test():
    window, regs, flags = roots(PROGRAM_HEX, [0] * 6 + [3, 262], [0] * 4)
    assert window.shape == (1, 120) and regs[0, 6, :2].tolist() == [1., 1.]
    assert window[0, :8].tolist() == [0., 0., 0., 1., 0., 0., 1., 0.]
    next_window = roots(PROGRAM_HEX, [0] * 8, [0] * 4, 7)[0]
    assert next_window[0, :8].tolist() == window[0, :8].tolist()
    return {"mechanics": "raw little-endian input and program-window ABI", "model_calls": 0,
            "previous_suffix_checks": _suffix["self_test"]()}


if __name__ == "__main__":
    print(self_test())
