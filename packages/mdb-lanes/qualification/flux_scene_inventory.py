"""Finite scene-factorial diagnostics and an independent chromatic pixel instrument.

No prompt parser or model is used to extract pixel components. Shape semantics
are deliberately left unknown. Contrast energy proposes supports, not causality.
"""
from __future__ import annotations

import itertools
from collections import deque


def make_plan():
    histories = []
    for shape, color, relation in itertools.product((0, 1), repeat=3):
        name = f"s{shape}c{color}r{relation}"
        prompt = (f"A red {('cube', 'sphere')[shape]} to the {('left', 'right')[relation]} of a "
                  f"{('blue', 'green')[color]} sphere on a plain white background. "
                  "Two separate objects, simple studio product photograph.")
        histories.append({"id": name, "factors": [shape, color, relation], "prompt": prompt})
    rows = [{"id": "native." + h["id"], "kind": "native", "history": h["id"]} for h in histories]
    rows.append({"id": "sham", "kind": "sham", "history": "s0c0r0"})
    rows += [{"id": factor + "." + mode, "kind": "patch", "factor": factor,
              "source": source, "history": "s0c0r0", "mode": mode}
             for factor, source in (("shapeA", "s1c0r0"), ("colorB", "s0c1r0"), ("relation", "s0c0r1"))
             for mode in ("full", "top", "shifted")]
    return {"histories": histories, "rows": rows}


def factorial_diagnostics(values):
    """Uniform binary Walsh coefficients; descriptive observed-state algebra."""
    import numpy as np

    keys = list(itertools.product((0, 1), repeat=3))
    if set(values) != set(keys):
        raise ValueError("a complete three-factor cube is required")
    arrays = [np.asarray(values[k], dtype=np.float64) for k in keys]
    if any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("finite matched tensor geometry required")
    coefficients, result = {}, {}
    for subset in itertools.product((0, 1), repeat=3):
        coefficient = sum(((-1) ** sum(1 - k[j] for j in range(3) if subset[j])) * a
                          for k, a in zip(keys, arrays)) / 8
        coefficients[subset] = coefficient
        if any(subset):
            label = "*".join(name for j, name in enumerate(("shapeA", "colorB", "relation")) if subset[j])
            result[label] = {"rms": float(np.sqrt(np.mean(coefficient ** 2))),
                             "l2": float(np.linalg.norm(coefficient.ravel()))}
    error = 0.
    for k, a in zip(keys, arrays):
        reconstructed = sum(((-1) ** sum(1 - k[j] for j in range(3) if subset[j])) * c
                            for subset, c in coefficients.items())
        error = max(error, float(np.max(np.abs(a - reconstructed))))
    return {"coefficients": result, "max_reconstruction_abs": error,
            "interpretation": "descriptive factorial of eight observed states; not independent causal operators"}


def propose_support(base, source, img_ids):
    """Top-quarter contrast energy; matched control shifts native grid addresses."""
    import numpy as np

    base, source, ids = (np.asarray(x) for x in (base, source, img_ids))
    if base.shape != source.shape or base.ndim != 3 or base.shape[0] != 1:
        raise ValueError("matched B1 image states required")
    ids = ids[0] if ids.ndim == 3 else ids
    if ids.shape != (base.shape[1], 4) or not np.isfinite(base).all() or not np.isfinite(source).all():
        raise ValueError("finite image states and native four-axis IDs required")
    # FLUX.2 IDs are (T,H,W,L). Do not infer spatial coordinates from row order.
    coordinates = [(int(p[1]), int(p[2])) for p in ids]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("this two-dimensional support requires unique H/W addresses")
    energy = np.mean((source[0].astype(np.float64) - base[0].astype(np.float64)) ** 2, axis=-1)
    count = max(1, len(energy) // 4)
    top = sorted(range(len(energy)), key=lambda i: (-energy[i], i))[:count]
    widths = sorted({p[1] for p in coordinates})
    lookup = {p: i for i, p in enumerate(coordinates)}
    shifted = []
    for i in top:
        h, w = coordinates[i]
        shifted.append(lookup[(h, widths[(widths.index(w) + len(widths) // 2) % len(widths)])])
    total = float(energy.sum())
    return {"top": top, "shifted": shifted, "energy": energy.tolist(), "coordinates": [list(p) for p in coordinates],
            "top_energy_fraction": float(energy[top].sum() / total) if total else None,
            "overlap_count": len(set(top) & set(shifted)), "rows": len(energy),
            "policy": "top quarter prefix-only image contrast; control horizontal half-grid cyclic shift; no final-image input"}


def pixel_inventory(rgb, *, chroma=.18, min_fraction=.001):
    """Generic colored connected components, numeric geometry, coarse hue labels.

    This is a purpose-built instrument for isolated colored primitives on white.
    It is neither open-world segmentation nor latent semantic-object extraction.
    """
    import numpy as np

    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("RGB uint8 pixels required")
    value = image.astype(np.float64) / 255
    mask = (value.max(-1) - value.min(-1) > chroma) & (value.max(-1) > .12)
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    components, label = [], 0
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        label += 1
        labels[y, x] = label
        queue, points = deque([(int(y), int(x))]), []
        while queue:
            py, px = queue.popleft()
            points.append((py, px))
            for ny, nx in ((py-1, px), (py+1, px), (py, px-1), (py, px+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = label
                    queue.append((ny, nx))
        if len(points) < min_fraction * height * width:
            continue
        coords = np.asarray(points)
        ys, xs = coords[:, 0], coords[:, 1]
        colors = np.median(value[ys, xs], axis=0)
        dominant = int(np.argmax(colors))
        ranked = np.sort(colors)
        hue = ("red", "green", "blue")[dominant] if ranked[-1] - ranked[-2] > .08 else "ambiguous"
        bounds = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
        components.append({"component_label": label, "color": hue, "median_rgb": colors.tolist(),
            "area": len(points), "area_fraction": len(points)/(height*width), "bounds": bounds,
            "centroid": [float(xs.mean()), float(ys.mean())],
            "extent": len(points)/((bounds[2]-bounds[0])*(bounds[3]-bounds[1])),
            "shape_semantics": "unknown", "identity_source": "pixel-connected chromatic region, not prompt or latent label"})
    components.sort(key=lambda c: (c["centroid"][0], c["centroid"][1]))
    relations = []
    for i, c in enumerate(components):
        c["id"] = f"component_{i:02d}"
        for j in range(i):
            relations.append({"subject": components[j]["id"], "predicate": "centroid-left-of", "object": c["id"],
                              "x_gap": c["centroid"][0] - components[j]["centroid"][0]})
    return {"components": components, "relations": relations,
            "instrument": {"chroma": chroma, "min_fraction": min_fraction, "connectivity": 4,
                           "scope": "colored visible components and image-plane centroid ordering; no 3D/occluded-object/shape inference"}}, labels
