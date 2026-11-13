"""Source-only contrast-dual writes and complete signed readout equations.

Axes combine two old FP32 even writes. They need not be even writes of one
realizable MLP input. Ideal neutrality and realized FP32 leakage are separate.
"""
from __future__ import annotations

import hashlib

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def readonly(value, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    require(np.isfinite(result).all(), "finite source array required")
    result.flags.writeable = False
    return result


def decompose(writes, consumer):
    require(set(writes) == {"plus", "minus"}, "exact old top/bottom write span required")
    require(all(np.asarray(value).dtype == np.float32 for value in writes.values()), "original once-cast FP32 writes required")
    P = np.stack([writes[name] for name in ("plus", "minus")], axis=1).astype(np.float64)
    c = readonly(consumer); require(P.ndim == 2 and c.shape == (P.shape[0],), "source span/reader shape differs")
    G, a = P.T @ P, P.T @ c
    U, singular, _ = np.linalg.svd(P, full_matrices=False)
    threshold = np.finfo(np.float64).eps*max(P.shape)*(singular[0] if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > threshold)); L = float(np.sqrt(np.trace(G)/2))
    projected = U[:, :rank] @ (U[:, :rank].T @ c)
    contrast_square = float(c @ projected)
    q1 = np.zeros(P.shape[0]); q0 = np.zeros(P.shape[0]); unit_coefficients = np.zeros(2)
    applicable = rank == 2 and float(a @ a) > 0
    if applicable:
        coefficients = np.linalg.solve(G, a); denominator = float(a @ coefficients)
        unit_coefficients = coefficients/denominator; q1 = P @ unit_coefficients
        q0 = P @ np.array([-a[1], a[0]])
    elif contrast_square > 0:
        q1 = projected/contrast_square
    elif rank:
        q0 = P[:, int(np.argmax(np.diag(G)))].copy()
    pivot = None
    if np.linalg.norm(q0) > 0:
        pivot = int(np.argmax(np.abs(q0)))
        if q0[pivot] < 0:
            q0 = -q0
    ideal = {name: readonly(np.zeros_like(value) if np.linalg.norm(value) == 0 else L*value/np.linalg.norm(value))
        for name, value in (("carrying", q1), ("neutral", q0))}
    realized = {name: readonly(value, np.float32) for name, value in ideal.items()}
    return dict(ideal=ideal, realized=realized, old_span=readonly(P), gram=readonly(G), contrast_coefficients=readonly(a),
        unit_contrast_q1=readonly(q1), unit_contrast_coefficients=readonly(unit_coefficients), raw_neutral_q0=readonly(q0),
        singular_values=readonly(singular), numeric_rank=rank, rank_threshold=float(threshold), common_norm=L,
        applicability="rank-two nonzero-contrast span" if applicable else "collapsed/contrast-degenerate axes retained",
        unit_q1_contrast=float(c @ q1), carrying_gain=float(c @ ideal["carrying"]), ideal_neutral_dot=float(c @ ideal["neutral"]),
        ideal_orthogonality=float(ideal["carrying"] @ ideal["neutral"]), neutral_sign_pivot=pivot,
        axis_diagnostics={name: dict(ideal_norm=float(np.linalg.norm(ideal[name])), realized_norm=float(np.linalg.norm(value.astype(np.float64))),
            realized_contrast_dot=float(c @ value.astype(np.float64)), cast_error_l2=float(np.linalg.norm(value.astype(np.float64)-ideal[name])),
            realized_span_residual_l2=float(np.linalg.norm(value.astype(np.float64)-U[:, :rank] @ (U[:, :rank].T @ value.astype(np.float64))))) for name, value in realized.items()},
        scope="Minimum-norm unit-contrast q1 is rescaled to commonL and no longer unit-contrast; ideal axes are orthogonal. Each receiving rho re-instantiates one fixed source rule; FP32 casts may leave the ideal span/null.")


def readout(base, decomposition, gamma, head, eps, cells, consumer_ids, *, collateral_ids=None):
    require(head.dtype == np.float32 and head.ndim == 2 and not head.flags.writeable and 0 < head.shape[0] < 2**32, "bound readonly FP32 full head required")
    base, gamma = readonly(base), readonly(gamma); hidden = head.shape[1]
    require(base.shape == gamma.shape == (hidden,) and eps > 0, "finite positive RMS readout geometry required")
    ideal, realized = decomposition["ideal"], decomposition["realized"]
    r1, r0 = (realized[name].astype(np.float64) for name in ("carrying", "neutral"))
    inputs = np.stack([gamma*base, gamma*r1, gamma*r0, gamma*ideal["carrying"], gamma*ideal["neutral"]], axis=1)
    full = np.empty((head.shape[0], 5), dtype=np.float64)
    for start in range(0, head.shape[0], 4096):
        block = head[start:start+4096].astype(np.float64)
        require(np.isfinite(block).all(), "finite original head required")
        full[start:start+len(block)] = block @ inputs
    coefficients, ideal_coefficients = full[:, :3], full[:, 3:]
    red, blue = consumer_ids; D0 = float(np.sqrt(np.mean(base*base)+eps)); baseline = coefficients[:, 0]/D0
    N0 = float(coefficients[red, 0]-coefficients[blue, 0]); rows = {}
    for cell in cells:
        key, alpha, beta = cell["id"], float(cell["carrying"]), float(cell["neutral"])
        require(key not in rows and np.isfinite(alpha) and np.isfinite(beta), "unique finite signed cell required")
        carrier = base+alpha*r1+beta*r0; denominator = float(np.sqrt(np.mean(carrier*carrier)+eps))
        numerator = coefficients[:, 0]+alpha*coefficients[:, 1]+beta*coefficients[:, 2]
        logits = numerator/denominator; order = np.argsort(-logits, kind="stable").astype(np.uint32)
        inverse = np.empty(head.shape[0], dtype=np.uint32); inverse[order] = np.arange(head.shape[0], dtype=np.uint32)
        ideal_carrier = base+alpha*ideal["carrying"]+beta*ideal["neutral"]
        ideal_D = float(np.sqrt(np.mean(ideal_carrier*ideal_carrier)+eps))
        ideal_N = N0+alpha*decomposition["carrying_gain"]
        delta = logits-baseline; shared_scale = D0/denominator
        displacement = (alpha*coefficients[:, 1]+beta*coefficients[:, 2])/denominator
        closure = delta-((shared_scale-1)*baseline+displacement)
        rows[key] = dict(carrying=alpha, neutral=beta, carrier=readonly(carrier), logits=readonly(logits), rank_order=readonly(order, np.uint32),
            greedy_token=int(order[0]), coefficient_greedy_token=int(np.argmax(numerator)),
            co_maximizers=readonly(np.flatnonzero(logits == logits.max()), np.uint32),
            numerator_co_maximizers=readonly(np.flatnonzero(numerator == numerator.max()), np.uint32),
            denominator=denominator, denominator_vs_baseline=denominator/D0, common_logit_scale=shared_scale,
            contrast_numerator=float(numerator[red]-numerator[blue]), ideal_contrast_numerator=ideal_N,
            realized_numerator_delta=float(numerator[red]-numerator[blue]-N0), ideal_numerator_delta=alpha*decomposition["carrying_gain"],
            realized_contrast=float(logits[red]-logits[blue]), ideal_gap=ideal_N/ideal_D,
            numerator_fixed_gap_counterfactual=N0/denominator, red_rank=int(inverse[red]), blue_rank=int(inverse[blue]),
            collateral_ranks={name: None if token is None else int(inverse[token]) for name, token in (collateral_ids or {}).items()},
            response=dict(orientation="cell-minus-own-baseline", elements=int(delta.size), sha256=hashlib.sha256(delta.astype("<f8").tobytes()).hexdigest(),
                rms=float(np.sqrt(np.mean(delta*delta))), signed_mean=float(np.mean(delta)), max_abs=float(np.max(np.abs(delta)))),
            scale_displacement_closure_maxabs=float(np.max(np.abs(closure))),
            scope="baseline" if alpha == beta == 0 else "numerator-neutral" if alpha == 0 else "contrast-carrying" if beta == 0 else "mixed")
    return dict(coefficients=readonly(coefficients), ideal_write_coefficients=readonly(ideal_coefficients), rows=rows,
        baseline_contrast_numerator=N0, baseline_denominator=D0, vocabulary=int(head.shape[0]),
        implications=["Pure ideal neutral doses preserve the supplied numerator and its pairwise sign; absolute gap scales with a positive RMS denominator.",
            "A common positive denominator cannot change the ideal vocabulary ranking. Neutral writes can change other-token numerators and hence the winner.",
            "Fixed FP32 payload casts and receiving-state additions may leak contrast; full finite logits, ranks and native discrepancies remain observations.",
            "Nine axial cells test signed carrying/neutral responses at two magnitudes; they do not test interactions or enumerate continuous2D winner regions."],
        arithmetic="Independent NumPy FP64 equations from source factors and fixed FP32 writes; complete original vocabulary; no native target supplies an equation RHS.")
