"""Independent NumPy FP64 spectral, parity and two-write readout equations.

No model, native runtime or file access. Original matrices are supplied explicitly;
the full vocabulary head remains read-only FP32 and is widened only in chunks.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

WEIGHTS = {"postnorm", "gate", "up", "down", "finalnorm"}
DOSES = (0., .125, .25, .5, 1.)


def readonly(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("finite original factors/state required")
    result.flags.writeable = False
    return result


def prepare(weights):
    if set(weights) != WEIGHTS:
        raise ValueError("exact bias-free RMS/gate/up/down/finalnorm factor names required")
    result = {name: readonly(value) for name, value in weights.items()}
    hidden = result["postnorm"].size
    if (result["postnorm"].shape != (hidden,) or not hidden or result["gate"].ndim != 2
            or result["gate"].shape[1] != hidden or not result["gate"].shape[0]
            or result["up"].shape != result["gate"].shape or result["down"].shape != (hidden, result["gate"].shape[0])
            or result["finalnorm"].shape != (hidden,)):
        raise ValueError("original factor geometry differs")
    return result


def vector(value, hidden):
    result = readonly(value)
    if result.shape != (hidden,):
        raise ValueError("one explicit hidden-state vector required")
    return result


def epsilon(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite original RMS epsilon required")
    return float(value)


def silu(value):
    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    result[positive] = value[positive] / (1. + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = value[~positive] * exponential / (1. + exponential)
    return result


def mlp(value, weights, eps):
    x = vector(value, weights["postnorm"].size); eps = epsilon(eps)
    denominator = float(np.sqrt(np.mean(x*x) + eps))
    normalized = weights["postnorm"] * (x / denominator)
    gate, up = weights["gate"] @ normalized, weights["up"] @ normalized
    product = silu(gate) * up
    return {name: readonly(row) for name, row in dict(input=x, postnorm=normalized, gate=gate, up=up,
        product=product, delta=weights["down"] @ product).items()}


def even_action(weights, consumer):
    """Weight-only B_c action; context norm/RMS denominator is not included."""
    gamma = weights["postnorm"]
    c = vector(consumer, gamma.size)
    downstream = readonly(weights["down"].T @ c)

    def action(value):
        z = gamma * vector(value, gamma.size)
        gate, up = weights["gate"] @ z, weights["up"] @ z
        return readonly(.25 * gamma * (weights["gate"].T @ (downstream * up) + weights["up"].T @ (downstream * gate)))

    return action


def lanczos(action, hidden, *, steps=128, seed=2718):
    """Deterministic matrix-free search; truncated extrema are strongest found."""
    if type(hidden) is not int or hidden < 1 or type(steps) is not int or steps < 1 or type(seed) is not int:
        raise ValueError("positive bounded spectral dimension/steps and integer seed required")
    count = min(hidden, steps)
    basis = np.empty((count, hidden), dtype=np.float64)
    q = np.random.default_rng(seed).standard_normal(hidden); q /= np.linalg.norm(q)
    diagonal, off_diagonal = [], []
    previous = np.zeros(hidden); beta = 0.
    for index in range(count):
        basis[index] = q
        raw = vector(action(q), hidden)
        z = np.array(raw - beta*previous, copy=True)
        alpha = float(q @ z); diagonal.append(alpha); z -= alpha*q
        # Twice-reorthogonalized original-factor Krylov basis, no dense B_c.
        for _ in range(2):
            z -= basis[:index+1].T @ (basis[:index+1] @ z)
        following = float(np.linalg.norm(z))
        if index + 1 == count or following <= np.finfo(np.float64).eps * hidden * max(float(np.linalg.norm(raw)), 1.):
            break
        off_diagonal.append(following); previous, q, beta = q, z/following, following
    basis = basis[:len(diagonal)]
    tridiagonal = np.diag(diagonal)
    if off_diagonal:
        tridiagonal += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, coefficients = np.linalg.eigh(tridiagonal)
    extremes = {}
    for name, index in (("plus", -1), ("minus", 0)):
        direction = basis.T @ coefficients[:, index]; direction /= np.linalg.norm(direction)
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0:
            direction = -direction
        applied = action(direction); rayleigh = float(direction @ applied)
        extremes[name] = dict(direction=readonly(direction), ritz_value=float(eigenvalues[index]), rayleigh_value=rayleigh,
            eigenaction_residual=float(np.linalg.norm(applied-rayleigh*direction)), sign_pivot=pivot)
    return dict(extremes=extremes, seed=seed, requested_steps=steps, searched_dimension=len(diagonal), hidden_dimension=hidden,
        complete_space_covered=len(diagonal) == hidden,
        orthogonality_maxabs=float(np.max(np.abs(basis @ basis.T-np.eye(len(diagonal))))),
        ritz_spectrum=readonly(eigenvalues), scope="Strongest found algebraic extrema in the declared weight-only Krylov space; no target selection or truncated global-extremum certificate")


def antipodal(direction, rho, weights, eps, consumer):
    hidden = weights["postnorm"].size
    direction = vector(direction, hidden)
    if type(rho) not in (int, float) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("positive finite observed raw cut norm required")
    if not np.isclose(np.linalg.norm(direction), 1., rtol=1e-12, atol=1e-12):
        raise ValueError("unit source direction required")
    positive = readonly(float(rho) * direction, np.float32)
    negative = readonly(-positive, np.float32)
    plus, minus = mlp(positive, weights, eps), mlp(negative, weights, eps)
    even, odd = (plus["delta"]+minus["delta"])/2, (plus["delta"]-minus["delta"])/2
    x = positive.astype(np.float64)
    normalized = weights["postnorm"] * x / np.sqrt(np.mean(x*x)+epsilon(eps))
    bilinear = .5 * (weights["down"] @ ((weights["gate"] @ normalized)*(weights["up"] @ normalized)))
    c = vector(consumer, hidden)
    return dict(positive_input=positive, negative_input=negative, positive=plus, negative=minus,
        even=readonly(even), odd=readonly(odd), even_bilinear=readonly(bilinear), write=readonly(even, np.float32),
        requested_raw_norm=float(rho), realized_raw_norm=float(np.linalg.norm(x)),
        consumer_even=float(c @ even), consumer_bilinear=float(c @ bilinear),
        parity_error=dict(max_abs=float(np.max(np.abs(even-bilinear))), rms=float(np.sqrt(np.mean((even-bilinear)**2)))),
        input_policy="One explicit FP32 cast of rho*direction; negative is its exact arithmetic antipode, including signed zeros")


def grid_forecast(base, writes, gamma, head, eps, *, doses=DOSES, chunk_rows=4096):
    """Full planes plus finite-grid logits; co-max regions use all vocabulary rows."""
    if not isinstance(head, np.ndarray) or head.dtype != np.float32 or head.ndim != 2 or head.flags.writeable or not head.size:
        raise ValueError("caller-bound read-only FP32 full vocabulary head required")
    if type(chunk_rows) is not int or not 1 <= chunk_rows <= 4096:
        raise ValueError("head widening chunks must contain1..4096 rows")
    hidden = head.shape[1]; eps = epsilon(eps)
    base, gamma = vector(base, hidden), vector(gamma, hidden)
    if set(writes) != {"plus", "minus"} or any(np.asarray(value).dtype != np.float32 for value in writes.values()):
        raise ValueError("exactly two fixed FP32 writes required")
    plus, minus = (vector(writes[name], hidden) for name in ("plus", "minus"))
    if (not isinstance(doses, (tuple, list)) or len(doses) != len(set(doses)) or not doses
            or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in doses) or 0 not in doses):
        raise ValueError("distinct finite0..1 doses including baseline0 required")
    coefficients = np.empty((head.shape[0], 3), dtype=np.float64)
    inputs = np.stack((gamma*base, gamma*plus, gamma*minus), axis=1)
    for start in range(0, head.shape[0], chunk_rows):
        block = np.array(head[start:start+chunk_rows], dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite original head row")
        coefficients[start:start+len(block)] = block @ inputs
    rows = {}
    baseline = coefficients[:, 0] / np.sqrt(np.mean(base*base)+eps)
    for alpha in doses:
        for beta in doses:
            key = format(float(alpha), ".17g") + "," + format(float(beta), ".17g")
            numerator = coefficients[:, 0] + alpha*coefficients[:, 1] + beta*coefficients[:, 2]
            carrier = base + alpha*plus + beta*minus
            normalizer = float(np.sqrt(np.mean(carrier*carrier)+eps)); logits = numerator/normalizer
            co_max = np.flatnonzero(numerator == numerator.max()).tolist()
            logit_co_max = np.flatnonzero(logits == logits.max()).tolist()
            effect = logits-baseline
            rows[key] = dict(alpha=float(alpha), beta=float(beta), logits=readonly(logits), carrier=readonly(carrier),
                normalizer=normalizer, greedy_token=min(logit_co_max), co_maximizers=co_max,
                logit_co_maximizers=logit_co_max, coefficient_greedy_token=min(co_max),
                response=dict(orientation="dose-minus-own0,0", elements=len(effect), dtype="float64",
                    sha256=hashlib.sha256(effect.astype("<f8").tobytes()).hexdigest(),
                    rms=float(np.sqrt(np.mean(effect*effect))), signed_mean=float(np.mean(effect)), max_abs=float(np.max(np.abs(effect)))))
    return dict(coefficients=readonly(coefficients), rows=rows, vocab_size=head.shape[0],
        regions=dict(domain=[[0., 1.], [0., 1.]], representation="Complete all-vocabulary halfspace family; no grid-only continuous-cell enumeration",
            inequality="For each candidate t and every vocabulary row v: (b_v-b_t)*alpha+(c_v-c_t)*beta <= a_t-a_v",
            coefficient_columns=["a", "b", "c"], row_identity="vocabulary token ID equals coefficient row index",
            boundary_policy="All satisfying rows co-maximize; lowest ID selects one greedy token"),
        arithmetic="Independent NumPy FP64 equations from original factors and fixed FP32 writes; native FP32 operation rounding remains an observation")


def payload_geometry(writes):
    matrix = np.stack([np.asarray(writes[name], dtype=np.float64) for name in ("plus", "minus")])
    gram = matrix @ matrix.T; eigenvalues = np.linalg.eigvalsh(gram)
    energy = float(np.trace(gram)); square = float(np.sum(eigenvalues*eigenvalues))
    return dict(gram=readonly(gram), gram_eigenvalues=readonly(eigenvalues),
        effective_dimension=None if not square else energy*energy/square,
        small_to_large=None if eigenvalues[-1] == 0 else max(0., float(eigenvalues[0]/eigenvalues[-1])),
        numeric_rank=int(np.linalg.matrix_rank(matrix)),
        scope="Rank/effective-dimension diagnostics only; two payload names do not establish a two-dimensional effect")


def select_futures(forecast):
    """Own-forecast-only baseline/region-interior selection with fixed fallback."""
    coefficients = forecast["coefficients"]; rows = forecast["rows"]
    candidates = {}
    for key, row in rows.items():
        token = row["coefficient_greedy_token"]
        point = np.array([row["alpha"], row["beta"]])
        gradient = coefficients[:, 1:]-coefficients[token, 1:]
        length = np.linalg.norm(gradient, axis=1)
        slack = coefficients[token, 0]-coefficients[:, 0]-gradient @ point
        active = length > 0
        distance = None if not active.any() else max(0., float(np.min(slack[active]/length[active])))
        box_distance = min(row["alpha"], 1-row["alpha"], row["beta"], 1-row["beta"])
        interior = box_distance if distance is None else min(box_distance, distance)
        score = (interior, float("inf") if distance is None else distance, -row["alpha"], -row["beta"])
        record = dict(row=key, token_id=token, region_distance=distance, box_distance=box_distance, interior_score=interior)
        if token not in candidates or score > candidates[token][0]:
            candidates[token] = (score, record)
    baseline = rows["0,0"]["coefficient_greedy_token"]
    selected, accounts = ["0,0"], [dict(row="0,0", reason="fixed-baseline", token_id=baseline)]
    ordered = sorted(((token, score, record) for token, (score, record) in candidates.items() if token != baseline),
        key=lambda value: (-value[1][0], -value[1][1], value[0]))
    for _, _, record in ordered[:3]:
        selected.append(record["row"]); accounts.append(dict(record, reason="distinct-own-predicted-winner-region"))
    for key in ("1,0", "0,1", "1,1"):
        if len(selected) == 4:
            break
        if key in rows and key not in selected:
            selected.append(key); accounts.append(dict(row=key, reason="fixed-corner-fallback", token_id=rows[key]["coefficient_greedy_token"]))
    return dict(rows=selected, accounts=accounts, predicted_regions=len(candidates), max_rows=4,
        authority="Only own presealed coefficient geometry; baseline then best-interior distinct winners, descending interior/distance then token ID; fixed corner fallback. No native outcome selects futures.")
