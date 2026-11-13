"""Algebraic metadata for anonymous native forward traces.

The module contains framework-free identities, exact def-use graph signatures,
and small numeric contrast helpers. Structural identities are kept separate
from empirical bindings: seeing an identity in this module does not assert that
a particular model implements or uses it.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from .forward_readable import ForwardReadableProgram

ALGEBRA_SCHEMA = "mdb-forward-algebra-v1"

NumberTree: TypeAlias = Real | Sequence[Any]
Cell: TypeAlias = tuple[Hashable, ...]


class ForwardAlgebraError(ValueError):
    """Raised when an algebraic candidate or numeric contrast is malformed."""


def _number(value: Any, label: str = "value") -> Real:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ForwardAlgebraError(f"{label} must contain real Python numbers")
    if not math.isfinite(float(value)):
        raise ForwardAlgebraError(f"{label} must contain finite numbers")
    return value


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _rebuild(template: Sequence[Any], values: list[Any]) -> list[Any] | tuple[Any, ...]:
    return tuple(values) if isinstance(template, tuple) else values


def _elementwise(left: Any, right: Any, operation: str, label: str = "value") -> Any:
    left_is_sequence, right_is_sequence = _sequence(left), _sequence(right)
    if left_is_sequence and right_is_sequence:
        if len(left) != len(right):
            raise ForwardAlgebraError(f"{label} sequences must have matching lengths")
        return _rebuild(
            left,
            [_elementwise(a, b, operation, label) for a, b in zip(left, right)],
        )
    if left_is_sequence:
        scalar = _number(right, label)
        return _rebuild(
            left, [_elementwise(item, scalar, operation, label) for item in left]
        )
    if right_is_sequence:
        scalar = _number(left, label)
        return _rebuild(
            right, [_elementwise(scalar, item, operation, label) for item in right]
        )

    a, b = _number(left, label), _number(right, label)
    if operation == "add":
        result = a + b
    elif operation == "sub":
        result = a - b
    elif operation == "mul":
        result = a * b
    else:  # pragma: no cover - internal call sites use the fixed operations above
        raise ForwardAlgebraError(f"unsupported numeric operation: {operation}")
    return _number(result, label)


def finite_difference(before: NumberTree, after: NumberTree) -> NumberTree:
    """Return ``after - before`` elementwise for scalars or nested lists/tuples."""
    return _elementwise(after, before, "sub", "finite difference")


def _scale(value: NumberTree, factor: Real, label: str) -> NumberTree:
    return _elementwise(value, _number(factor, label), "mul", label)


def factorial_contrast(
    observations: Mapping[Cell, NumberTree],
    *,
    factors: Sequence[str],
    contrast_factors: Sequence[str] | None = None,
    levels: tuple[Hashable, Hashable] = (0, 1),
) -> dict[str, Any]:
    """Compute an average marginal finite-difference contrast on a complete 2^k grid.

    ``factors`` defines the cell-key order. The selected factors receive the
    usual low=-1/high=+1 sign product; unselected factors are averaged. Thus a
    two-factor contrast over a 2x2 table is the unscaled difference of
    differences, while a main effect in a larger design is averaged over the
    other factors. Coefficients and the averaging denominator are returned so
    the convention is explicit and reproducible.
    """
    if isinstance(factors, str):
        raise ForwardAlgebraError("factors must be a sequence of distinct names, not one string")
    factor_names = tuple(factors)
    if (not factor_names or len(factor_names) > 12
            or any(not isinstance(name, str) or not name for name in factor_names)
            or len(set(factor_names)) != len(factor_names)):
        raise ForwardAlgebraError("factors must be 1 to 12 distinct non-empty names")
    if (not isinstance(levels, tuple) or len(levels) != 2
            or levels[0] == levels[1]):
        raise ForwardAlgebraError("levels must be a pair of distinct labels")
    try:
        hash(levels[0])
        hash(levels[1])
    except TypeError as exc:
        raise ForwardAlgebraError("factor levels must be hashable") from exc

    if isinstance(contrast_factors, str):
        raise ForwardAlgebraError("contrast_factors must be a sequence of names, not one string")
    selected = factor_names if contrast_factors is None else tuple(contrast_factors)
    if (not selected or len(set(selected)) != len(selected)
            or any(name not in factor_names for name in selected)):
        raise ForwardAlgebraError("contrast_factors must select known factors")

    expected_cells = tuple(itertools.product(levels, repeat=len(factor_names)))
    actual_cells = set(observations)
    if actual_cells != set(expected_cells):
        missing = sorted(set(expected_cells) - actual_cells, key=repr)
        extra = sorted(actual_cells - set(expected_cells), key=repr)
        raise ForwardAlgebraError(
            f"observations must contain the complete factorial grid; missing={missing}, extra={extra}"
        )

    selected_indices = {factor_names.index(name) for name in selected}
    coefficient_rows: list[dict[str, Any]] = []
    total: NumberTree | None = None
    for cell in expected_cells:
        coefficient = 1
        for index in selected_indices:
            coefficient *= -1 if cell[index] == levels[0] else 1
        coefficient_rows.append({"cell": list(cell), "coefficient": coefficient})
        term = _scale(observations[cell], coefficient, "factorial contrast")
        total = term if total is None else _elementwise(
            total, term, "add", "factorial contrast"
        )

    assert total is not None  # expected_cells has at least one member
    denominator = 2 ** (len(factor_names) - len(selected_indices))
    value = total if denominator == 1 else _scale(
        total, 1.0 / denominator, "factorial contrast"
    )
    return {
        "schema": ALGEBRA_SCHEMA,
        "kind": "factorial-contrast",
        "factors": list(factor_names),
        "contrast_factors": list(selected),
        "levels": list(levels),
        "coefficients": coefficient_rows,
        "averaging_denominator": denominator,
        "value": value,
        "interpretation_limit": (
            "a sampled signed corner contrast; it does not establish continuous multilinearity "
            "or a behavioral law"
        ),
        "provenance": "empirical-observation",
    }


def mobius_corner_coefficients(
    observations: Mapping[Cell, NumberTree],
    *,
    factors: Sequence[str],
    levels: tuple[Hashable, Hashable] = (0, 1),
) -> dict[str, Any]:
    """Return the unique 0/1 multilinear interpolant coefficients for sampled corners.

    This is a change of basis over the observed corner table, not evidence that
    an unknown function is multilinear between the corners.
    """
    if isinstance(factors, str):
        raise ForwardAlgebraError("factors must be a sequence of distinct names, not one string")
    factor_names = tuple(factors)
    if (not factor_names or len(factor_names) > 12
            or any(not isinstance(name, str) or not name for name in factor_names)
            or len(set(factor_names)) != len(factor_names)):
        raise ForwardAlgebraError("factors must be 1 to 12 distinct non-empty names")
    if (not isinstance(levels, tuple) or len(levels) != 2
            or levels != (0, 1)):
        raise ForwardAlgebraError("Möbius corner coefficients require levels (0, 1)")
    expected = set(itertools.product(levels, repeat=len(factor_names)))
    actual = set(observations)
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise ForwardAlgebraError(
            f"observations must contain the complete 0/1 corner grid; missing={missing}, extra={extra}"
        )

    terms: list[dict[str, Any]] = []
    for size in range(len(factor_names) + 1):
        for selected in itertools.combinations(range(len(factor_names)), size):
            coefficient: NumberTree | None = None
            for mask in itertools.product((0, 1), repeat=size):
                corner = [0] * len(factor_names)
                for index, level in zip(selected, mask):
                    corner[index] = level
                sign = -1 if (size - sum(mask)) % 2 else 1
                term = _scale(observations[tuple(corner)], sign, "Möbius coefficient")
                coefficient = term if coefficient is None else _elementwise(
                    coefficient, term, "add", "Möbius coefficient"
                )
            assert coefficient is not None
            terms.append({
                "factors": [factor_names[index] for index in selected],
                "coefficient": coefficient,
            })
    return {
        "schema": ALGEBRA_SCHEMA,
        "kind": "corner-mobius-coefficients",
        "factors": list(factor_names),
        "levels": [0, 1],
        "terms": terms,
        "provenance": "empirical-observation",
        "interpretation_limit": (
            "these coefficients define the unique multilinear interpolant through the sampled "
            "corners; they do not establish that the underlying function is multilinear"
        ),
    }


def multilinear_predict(
    coefficients: Mapping[Any, Any],
    *,
    factors: Sequence[str],
    coordinates: Mapping[str, Real],
) -> NumberTree:
    """Evaluate a 0/1 multilinear polynomial from Möbius coefficients."""
    if isinstance(factors, str):
        raise ForwardAlgebraError("factors must be a sequence of distinct names, not one string")
    factor_names = tuple(factors)
    term_order = tuple(
        subset
        for size in range(len(factor_names) + 1)
        for subset in itertools.combinations(factor_names, size)
    )
    expected_terms = set(term_order)
    if (not factor_names or len(set(factor_names)) != len(factor_names)
            or any(not isinstance(name, str) or not name for name in factor_names)):
        raise ForwardAlgebraError("factors must be distinct non-empty names")
    if set(coefficients) == expected_terms:
        coefficient_map = coefficients
    elif "terms" in coefficients and isinstance(coefficients["terms"], list):
        if tuple(coefficients.get("factors", ())) != factor_names:
            raise ForwardAlgebraError("Möbius coefficient factor order does not match factors")
        coefficient_map = {}
        for row in coefficients["terms"]:
            if (not isinstance(row, Mapping) or set(row) != {"factors", "coefficient"}
                    or not isinstance(row["factors"], list)):
                raise ForwardAlgebraError("Möbius coefficient document has a malformed term")
            subset = tuple(row["factors"])
            if subset in coefficient_map:
                raise ForwardAlgebraError("Möbius coefficient document has duplicate terms")
            coefficient_map[subset] = row["coefficient"]
        if set(coefficient_map) != expected_terms:
            raise ForwardAlgebraError("coefficients must contain every factor subset exactly once")
    else:
        raise ForwardAlgebraError("coefficients must contain exactly one term for every factor subset")
    if set(coordinates) != set(factor_names):
        raise ForwardAlgebraError("coordinates must provide exactly the requested factors")
    values = {name: _number(coordinates[name], f"coordinate {name}") for name in factor_names}
    if any(value < 0 or value > 1 for value in values.values()):
        raise ForwardAlgebraError("multilinear interpolation coordinates must lie in [0, 1]")

    prediction: NumberTree | None = None
    for subset in term_order:
        weight: Real = 1
        for name in subset:
            weight *= values[name]
        term = _scale(coefficient_map[subset], weight, "multilinear prediction")
        prediction = term if prediction is None else _elementwise(
            prediction, term, "add", "multilinear prediction"
        )
    assert prediction is not None
    return prediction


def factorial_contrasts(
    observations: Mapping[Cell, NumberTree],
    *,
    factors: Sequence[str],
    levels: tuple[Hashable, Hashable] = (0, 1),
) -> dict[str, Any]:
    """Return all non-empty factorial contrasts and, for 0/1, corner Möbius coefficients."""
    factor_names = tuple(factors)
    contrasts = []
    for size in range(1, len(factor_names) + 1):
        for subset in itertools.combinations(factor_names, size):
            contrasts.append(factorial_contrast(
                observations,
                factors=factor_names,
                contrast_factors=subset,
                levels=levels,
            ))
    result = {
        "schema": ALGEBRA_SCHEMA,
        "kind": "factorial-contrast-set",
        "factors": list(factor_names),
        "contrasts": contrasts,
        "provenance": "empirical-observation",
        "interpretation_limit": (
            "contrast coefficients summarize sampled cells and do not establish continuous "
            "multilinearity or a behavioral law"
        ),
    }
    if levels == (0, 1):
        result["corner_mobius"] = mobius_corner_coefficients(
            observations, factors=factor_names, levels=(0, 1)
        )
    return result


@dataclass(frozen=True)
class AffineScan:
    """One element ``(D, W)`` representing ``x -> D*x + W``."""

    decay: NumberTree
    write: NumberTree


def compose_affine_scan(later: AffineScan, earlier: AffineScan) -> AffineScan:
    """Compose affine scan elements in application order: later after earlier."""
    if not isinstance(later, AffineScan) or not isinstance(earlier, AffineScan):
        raise ForwardAlgebraError("scan composition requires two AffineScan values")
    return AffineScan(
        decay=_elementwise(later.decay, earlier.decay, "mul", "scan decay composition"),
        write=_elementwise(
            later.write,
            _elementwise(later.decay, earlier.write, "mul", "scan write composition"),
            "add",
            "scan write composition",
        ),
    )


def apply_affine_scan(scan: AffineScan, state: NumberTree) -> NumberTree:
    """Apply an affine scan element to a scalar or elementwise numeric state."""
    if not isinstance(scan, AffineScan):
        raise ForwardAlgebraError("scan application requires an AffineScan value")
    return _elementwise(
        _elementwise(scan.decay, state, "mul", "scan application"),
        scan.write,
        "add",
        "scan application",
    )


def log_clock_residual(left: Real, right: Real, product: Real | None = None) -> float:
    """Evaluate ``log(product)-log(left)-log(right)`` on the positive domain."""
    a, b = _number(left, "log-clock left factor"), _number(right, "log-clock right factor")
    joined = _number(a * b if product is None else product, "log-clock product")
    if a <= 0 or b <= 0 or joined <= 0:
        raise ForwardAlgebraError("log-clock factors and product must be positive")
    return math.log(float(joined)) - math.log(float(a)) - math.log(float(b))


def bilinear_form(left: Sequence[Real], matrix: Sequence[Sequence[Real]],
                  right: Sequence[Real]) -> Real:
    """Evaluate ``leftᵀ matrix right`` using ordinary Python numeric sequences."""
    if not _sequence(left) or not _sequence(right) or not _sequence(matrix):
        raise ForwardAlgebraError("bilinear form inputs must be sequences")
    if len(matrix) != len(left):
        raise ForwardAlgebraError("bilinear matrix row count must match the left vector")
    result: Real = 0
    for row_index, row in enumerate(matrix):
        if not _sequence(row) or len(row) != len(right):
            raise ForwardAlgebraError("bilinear matrix column count must match the right vector")
        left_value = _number(left[row_index], "bilinear left vector")
        for column_index, entry in enumerate(row):
            result += left_value * _number(entry, "bilinear matrix") * _number(
                right[column_index], "bilinear right vector"
            )
    return _number(result, "bilinear result")


def bilinear_factorial_identity(
    left_low: Sequence[Real],
    left_high: Sequence[Real],
    right_low: Sequence[Real],
    right_high: Sequence[Real],
    matrix: Sequence[Sequence[Real]],
) -> dict[str, Any]:
    """Evaluate the exact mixed-difference identity for a supplied bilinear form."""
    x0, x1 = list(left_low), list(left_high)
    y0, y1 = list(right_low), list(right_high)
    cells = {
        (0, 0): bilinear_form(x0, matrix, y0),
        (1, 0): bilinear_form(x1, matrix, y0),
        (0, 1): bilinear_form(x0, matrix, y1),
        (1, 1): bilinear_form(x1, matrix, y1),
    }
    interaction = factorial_contrast(
        cells, factors=("left", "right"), levels=(0, 1)
    )["value"]
    expected = bilinear_form(
        finite_difference(x0, x1), matrix, finite_difference(y0, y1)
    )
    return {
        "schema": ALGEBRA_SCHEMA,
        "kind": "bilinear-factorial-identity",
        "cell_values": {f"{left}{right}": value for (left, right), value in cells.items()},
        "mixed_difference": interaction,
        "bilinear_cross_term": expected,
        "remainder": finite_difference(expected, interaction),
        "provenance": "structural-identity",
        "numeric_evaluation": "computed-from-supplied-vectors-and-matrix",
    }


def bilinear_interaction_remainder(
    observed_outputs: Mapping[Cell, NumberTree],
    bilinear_cross_term: NumberTree,
    *,
    factors: tuple[str, str] = ("left", "right"),
    levels: tuple[Hashable, Hashable] = (0, 1),
) -> dict[str, Any]:
    """Subtract a supplied bilinear cross-term from an observed 2-factor interaction.

    The remainder records what the supplied bilinear term does not explain. It
    can include nonlinear response, other paths, and numeric effects; it is not
    automatically a uniquely identified nonlinear component.
    """
    contrast = factorial_contrast(
        observed_outputs,
        factors=factors,
        levels=levels,
    )
    remainder = finite_difference(bilinear_cross_term, contrast["value"])
    return {
        "schema": ALGEBRA_SCHEMA,
        "kind": "bilinear-interaction-remainder",
        "observed_interaction": contrast["value"],
        "bilinear_cross_term": bilinear_cross_term,
        "nonlinear_remainder": remainder,
        "remainder_scope": (
            "observed interaction minus supplied bilinear term; may include nonlinear response, "
            "omitted paths, and numerical effects"
        ),
        "contrast": contrast,
        "provenance": "empirical-observation",
    }


def structural_identities() -> list[dict[str, Any]]:
    """Return reusable algebraic candidates, independent of any model trace."""
    return [
        {
            "id": "affine-scan-compose-v1",
            "kind": "affine-scan-monoid",
            "state_map": "x -> D*x + W",
            "expression": "(D2,W2) ∘ (D1,W1) = (D2*D1, W2 + D2*W1)",
            "order": "apply (D1,W1), then (D2,W2)",
            "provenance": "structural-identity",
        },
        {
            "id": "positive-log-character-v1",
            "kind": "additive-log-clock-character",
            "expression": "log(a*b) = log(a) + log(b)",
            "domain": ["a > 0", "b > 0"],
            "provenance": "structural-identity",
        },
        {
            "id": "bilinear-reader-interaction-v1",
            "kind": "bilinear-mixed-difference",
            "expression": (
                "B(x1,y1) - B(x1,y0) - B(x0,y1) + B(x0,y0) "
                "= B(x1-x0, y1-y0)"
            ),
            "assumption": "B is linear in each argument with the other held fixed",
            "provenance": "structural-identity",
        },
    ]


def _program_index(program: ForwardReadableProgram) -> tuple[
    dict[int, dict[str, Any]], dict[str, dict[str, Any]]
]:
    if not isinstance(program.fingerprint, str):
        raise ForwardAlgebraError("readable program is missing its trace fingerprint")
    document = program.program()
    events = {
        event["sequence"]: event
        for module in document["modules"]
        for event in module["events"]
    }
    values = {row["value_id"]: row for row in document["values"]}
    return events, values


def _native_argument(program: ForwardReadableProgram, sequence: int, index: int,
                     name: str, default: Any = None) -> Any:
    """Read one sealed native argument to preserve scalar operator options."""
    event = program.trace["events"][sequence]
    arguments = event["arguments"]
    if isinstance(arguments, Mapping) and "args" in arguments and "kwargs" in arguments:
        packed = arguments.get("args")
        if isinstance(packed, Mapping) and set(packed) == {"$tuple"}:
            args = packed["$tuple"]
        elif isinstance(packed, list):
            args = packed
        elif packed is None:
            args = []
        else:
            args = [packed]
        kwargs = arguments.get("kwargs") or {}
        if name in kwargs:
            return kwargs[name]
        return args[index] if index < len(args) else default
    if isinstance(arguments, Mapping):
        return arguments.get(name, default)
    return default


def graph_candidates(program: ForwardReadableProgram) -> list[dict[str, Any]]:
    """Extract exact, role-neutral motifs from a ``ForwardReadableProgram``.

    A motif reports only native operator and def-use facts. The ``affine_views``
    for a product-plus-addend motif enumerate both possible multiplication
    operand roles instead of guessing which value is a state or coefficient.
    """
    events, values = _program_index(program)
    result: list[dict[str, Any]] = []

    for add_sequence, add_event in events.items():
        if not add_event["operator"].startswith("aten.add.") or len(add_event["inputs"]) != 2:
            continue
        alpha = _native_argument(program, add_sequence, 2, "alpha", 1)
        for product_index, product_value in enumerate(add_event["inputs"]):
            product_sequence = values[product_value]["producer"]
            product_event = events.get(product_sequence)
            if (product_event is None
                    or not product_event["operator"].startswith("aten.mul.")
                    or len(product_event["inputs"]) != 2):
                continue
            addend = add_event["inputs"][1 - product_index]
            left, right = product_event["inputs"]
            outputs = [row["value_id"] for row in add_event["outputs"]]
            result.append({
                "id": f"graph-product-plus-addend-e{product_sequence}-e{add_sequence}",
                "kind": "product-plus-addend",
                "expression": add_event["expression"],
                "product_expression": product_event["expression"],
                "product_inputs": [left, right],
                "addend_value": addend,
                "output_values": outputs,
                "affine_views": ([
                    {"coefficient_value": left, "variable_value": right,
                     "offset_value": addend},
                    {"coefficient_value": right, "variable_value": left,
                     "offset_value": addend},
                ] if alpha == 1 else []),
                "add_alpha": alpha,
                "affine_views_omitted_reason": (
                    None if alpha == 1 else "native add alpha changes the operand coefficient"
                ),
                "events": [product_sequence, add_sequence],
                "related_identity_ids": ["affine-scan-compose-v1"],
                "provenance": "structural-observation",
            })

    for sum_sequence, sum_event in events.items():
        if sum_event["operator"] not in {"aten.sum.default", "aten.sum.dim_IntList"}:
            continue
        if not sum_event["inputs"]:
            continue
        product_value = sum_event["inputs"][0]
        product_sequence = values[product_value]["producer"]
        product_event = events.get(product_sequence)
        if (product_event is None
                or not product_event["operator"].startswith("aten.mul.")
                or len(product_event["inputs"]) != 2):
            continue
        result.append({
            "id": f"graph-sum-of-product-e{product_sequence}-e{sum_sequence}",
            "kind": "sum-of-product-contraction",
            "expression": sum_event["expression"],
            "product_expression": product_event["expression"],
            "factor_values": list(product_event["inputs"]),
            "product_value": product_value,
            "output_values": [row["value_id"] for row in sum_event["outputs"]],
            "events": [product_sequence, sum_sequence],
            "related_identity_ids": ["bilinear-reader-interaction-v1"],
            "provenance": "structural-observation",
        })

    matrix_operators = {"aten.matmul.default", "aten.mm.default", "aten.bmm.default"}
    for pattern in program.list_patterns():
        if pattern["kind"] not in {
            "softmax-weighted-matrix-product", "softmax-weighted-reduction"
        }:
            continue
        weight_value = pattern.get("weight_value")
        if weight_value not in values:
            continue
        softmax_sequence = values[weight_value]["producer"]
        softmax_event = events.get(softmax_sequence)
        if (softmax_event is None
                or not softmax_event["operator"].startswith("aten.softmax.")
                or not softmax_event["inputs"]):
            continue
        score_value = softmax_event["inputs"][0]
        score_sequence = values[score_value]["producer"]
        score_event = events.get(score_sequence)
        if score_event is None or score_event["operator"] not in matrix_operators:
            continue
        output_values = pattern["outputs"]
        read_sequence = values[output_values[0]]["producer"] if output_values else None
        event_ids = sorted({
            sequence for sequence in [score_sequence, softmax_sequence, read_sequence]
            if sequence is not None
        })
        result.append({
            "id": "graph-softmax-bilinear-read-" + "-".join(map(str, event_ids)),
            "kind": "softmax-weighted-bilinear-read",
            "score": {
                "expression": score_event["expression"],
                "input_values": list(score_event["inputs"]),
                "output_values": [row["value_id"] for row in score_event["outputs"]],
                "event": score_sequence,
            },
            "weights": {
                "expression": softmax_event["expression"],
                "input_value": score_value,
                "output_value": weight_value,
                "event": softmax_sequence,
            },
            "read": {
                "expression": pattern["expression"],
                "payload_value": pattern.get("other_value"),
                "output_values": list(output_values),
                "events": list(pattern["events"]),
            },
            "events": event_ids,
            "related_identity_ids": ["bilinear-reader-interaction-v1"],
            "provenance": "structural-observation",
        })

    for sequence, event in events.items():
        if not event["operator"].startswith("aten.log.") or not event["inputs"]:
            continue
        product_value = event["inputs"][0]
        product_sequence = values[product_value]["producer"]
        product_event = events.get(product_sequence)
        if (product_event is None
                or not product_event["operator"].startswith("aten.mul.")
                or len(product_event["inputs"]) != 2):
            continue
        result.append({
            "id": f"graph-log-of-product-e{product_sequence}-e{sequence}",
            "kind": "log-of-product",
            "observed_expression": event["expression"],
            "product_expression": product_event["expression"],
            "factor_values": list(product_event["inputs"]),
            "output_values": [row["value_id"] for row in event["outputs"]],
            "equivalent_expression": "log(a) + log(b), when a > 0 and b > 0",
            "events": [product_sequence, sequence],
            "related_identity_ids": ["positive-log-character-v1"],
            "provenance": "structural-observation",
        })

    for pattern in program.list_patterns():
        if pattern["kind"] != "exp-of-softplus-scaled-negative-exp":
            continue
        result.append({
            "id": "graph-softplus-negative-exp-" + "-".join(map(str, pattern["events"])),
            "kind": "exp-softplus-negative-exp",
            "expression": pattern["expression"],
            "events": list(pattern["events"]),
            "values": list(pattern["values"]),
            "conditional_identity": {
                "expression": "0 < exp(softplus(t) * (-exp(a))) < 1",
                "arithmetic_scope": "real arithmetic",
                "conditions": [
                    "finite inputs",
                    "the recorded softplus configuration yields a positive output",
                ],
                "native_numeric_caveat": (
                    "finite-precision execution may underflow to 0 or round to 1; inspect "
                    "recorded native values separately"
                ),
            },
            "provenance": "structural-observation",
        })

    return result


def acquire_symbol_candidates(program: ForwardReadableProgram) -> dict[str, Any]:
    """Return generic identities and trace-bound graph signatures, with no findings merged in."""
    return {
        "schema": ALGEBRA_SCHEMA,
        "trace_fingerprint": program.fingerprint,
        "structural_identities": structural_identities(),
        "graph_candidates": graph_candidates(program),
        "empirical_bindings": [],
        "empirical_binding_policy": "store numeric observations separately by candidate ID and trace scope",
    }


def bind_empirical_observation(
    candidate_id: str,
    *,
    trace_fingerprint: str,
    scope: Mapping[str, Any],
    observed: NumberTree,
    expected: NumberTree,
    evidence_refs: Sequence[str],
) -> dict[str, Any]:
    """Create a separate trace-scoped empirical binding for a structural candidate."""
    if (not isinstance(candidate_id, str) or not candidate_id
            or not isinstance(trace_fingerprint, str)
            or len(trace_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in trace_fingerprint)):
        raise ForwardAlgebraError("candidate ID and lowercase trace fingerprint are required")
    if not isinstance(scope, Mapping) or not scope:
        raise ForwardAlgebraError("empirical binding scope must be a non-empty mapping")
    refs = list(evidence_refs)
    if not refs or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
        raise ForwardAlgebraError("empirical binding requires non-empty evidence references")
    return {
        "schema": ALGEBRA_SCHEMA,
        "candidate_id": candidate_id,
        "trace_fingerprint": trace_fingerprint,
        "scope": dict(scope),
        "observed": observed,
        "expected": expected,
        "residual": finite_difference(expected, observed),
        "evidence_refs": refs,
        "provenance": "empirical-observation",
    }


__all__ = [
    "ALGEBRA_SCHEMA", "AffineScan", "ForwardAlgebraError", "acquire_symbol_candidates",
    "apply_affine_scan", "bilinear_factorial_identity", "bilinear_form",
    "bilinear_interaction_remainder", "bind_empirical_observation", "compose_affine_scan",
    "factorial_contrast", "factorial_contrasts", "finite_difference", "graph_candidates",
    "log_clock_residual", "mobius_corner_coefficients", "multilinear_predict",
    "structural_identities",
]
