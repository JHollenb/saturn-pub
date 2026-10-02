import copy
import json
from pathlib import Path

import pytest

from mdb.hyperion import (
    HYPERION_REPORT_SCHEMA,
    HyperionError,
    HyperionObservation,
    adapt_ar_induction_report,
    adapt_ird_depth_report,
    adapt_ird_kernel_report,
    adapt_ird_path_report,
    compile_dynamic_circuit,
    verify_hyperion_report,
)

CONSUMER = "flux2-klein-native-render"


def _observation(
    cell: str,
    arm: str,
    schedule: dict[int, float],
    effect: float,
    *,
    phase: str = "discovery",
    endpoint: str | None = None,
) -> HyperionObservation:
    target, seed = cell.split("-")
    return HyperionObservation(
        observation_id=f"{phase}:{cell}:{arm}",
        cell_id=cell,
        arm_id=arm,
        phase=phase,
        arm_kind="candidate" if phase != "control" else "noop",
        sites=("conditioner.subject-row",),
        schedule=tuple(schedule.items()),
        target_effect=effect,
        consumer_id=CONSUMER,
        evidence_ref=f"fixture.json#/{cell}/{arm}",
        endpoint_equivalence_id=endpoint,
        metadata={"target": target, "seed": int(seed)},
    )


def _dynamic_rows() -> list[HyperionObservation]:
    rows: list[HyperionObservation] = []
    for cell, scale in (("lion-1", 1.0), ("wolf-2", 1.2)):
        effects = {
            "s0": 0.18 * scale,
            "s1": 0.16 * scale,
            "s2": 0.05 * scale,
            "s3": 0.02 * scale,
            "s01": 0.42 * scale,
            "s23": 0.08 * scale,
            "all": 0.68 * scale,
            "all_half": 0.32 * scale,
            "s23_double": 0.18 * scale,
        }
        schedules = {
            "s0": {0: 1.0},
            "s1": {1: 1.0},
            "s2": {2: 1.0},
            "s3": {3: 1.0},
            "s01": {0: 1.0, 1: 1.0},
            "s23": {2: 1.0, 3: 1.0},
            "all": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
            "all_half": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
            "s23_double": {2: 2.0, 3: 2.0},
        }
        rows.extend(
            _observation(cell, arm, schedules[arm], effect) for arm, effect in effects.items()
        )
        rows.append(
            _observation(
                cell,
                "pos01_neg23",
                {0: 1.0, 1: 1.0, 2: -1.0, 3: -1.0},
                0.34 * scale,
                phase="heldout",
                endpoint="net-zero",
            )
        )
        rows.append(
            _observation(
                cell,
                "neg01_pos23",
                {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0},
                -0.30 * scale,
                phase="heldout",
                endpoint="net-zero",
            )
        )
        rows.append(_observation(cell, "noop", {}, 0.0, phase="control"))
    return rows


def _compile(rows: list[HyperionObservation]) -> dict:
    return compile_dynamic_circuit(
        rows,
        symbol_id="flux2.animal.identity.ird",
        logical_symbol="animal.identity",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        consumer_id=CONSUMER,
        time_axis="denoising-consumption-step",
        address_space="conditioner-content",
        residual_definition="target minus base at the subject conditioner row",
        source_artifacts=[{"path": "fixture.json", "sha256": "0" * 64}],
        context={"steps": 4, "guidance": 1.0},
    )


def test_compiler_identifies_weighted_temporal_integral() -> None:
    report = _compile(_dynamic_rows())

    assert report["schema"] == HYPERION_REPORT_SCHEMA
    assert report["classification"]["primary"] == "weighted-temporal-integral"
    assert "path-order-sensitive" in report["classification"]["properties"]
    assert "dose-nonexchangeable-across-time" in report["classification"]["properties"]
    assert report["symbol"]["terminal_claim_status"] == "not_assessed"
    assert report["symbol"]["evidence"]["level"] == "convergent_trend"
    assert report["symbol"]["writer"]["executable"] is False
    assert report["symbol"]["context_fingerprint"]
    assert verify_hyperion_report(report)["passed"] is True


def test_verifier_detects_report_and_symbol_tampering() -> None:
    report = _compile(_dynamic_rows())
    tampered = copy.deepcopy(report)
    tampered["classification"]["primary"] = "point-local"
    with pytest.raises(HyperionError, match="report fingerprint"):
        verify_hyperion_report(tampered)

    symbol_tampered = copy.deepcopy(report)
    symbol_tampered["symbol"]["logical_symbol"] = "animal.color"
    body = {key: value for key, value in symbol_tampered.items() if key != "fingerprint"}
    from mdb.hyperion import _digest

    symbol_tampered["fingerprint"] = _digest(body)
    with pytest.raises(HyperionError, match="symbol fingerprint"):
        verify_hyperion_report(symbol_tampered)

    compiler_tampered = copy.deepcopy(report)
    compiler_tampered["compiler"]["implementation_sha256"] = "f" * 64
    body = {key: value for key, value in compiler_tampered.items() if key != "fingerprint"}
    compiler_tampered["fingerprint"] = _digest(body)
    with pytest.raises(HyperionError, match="implementation drifted"):
        verify_hyperion_report(compiler_tampered)


def test_observation_rejects_duplicate_schedule_times() -> None:
    with pytest.raises(HyperionError, match="unique"):
        HyperionObservation(
            observation_id="duplicate-time",
            cell_id="lion-1",
            arm_id="bad",
            phase="discovery",
            arm_kind="candidate",
            sites=("site",),
            schedule=((0, 1.0), (0, 2.0)),
            target_effect=1.0,
            consumer_id=CONSUMER,
            evidence_ref="fixture",
        )


def test_ird_adapters_keep_path_rows_heldout(tmp_path: Path) -> None:
    kernel_payload = {
        "schema": "saturn-ird-p4-kernel-v1",
        "execution_receipt": {
            "mrun_job_id": "job-k",
            "mrun_ok": True,
            "report_present": True,
        },
        "kernel": {
            "branches": [
                {
                    "target": "lion",
                    "seed": 1,
                    "arm": "s0",
                    "alpha": 1.0,
                    "sched": [0],
                    "progress": 0.2,
                    "calls": 4,
                }
            ],
            "gates": {"noop_1": 0.0},
        },
    }
    kernel_path = tmp_path / "kernel.json"
    kernel_path.write_text(json.dumps(kernel_payload), encoding="utf-8")
    kernel_rows, kernel_source = adapt_ird_kernel_report(
        kernel_payload, path=kernel_path, consumer_id=CONSUMER
    )

    path_payload = {
        "schema": "saturn-ird-allmodels-p1-v1",
        "execution_receipt": {
            "mrun_job_id": "job-p",
            "mrun_ok": True,
            "report_present": True,
        },
        "p1": {
            "branches": [
                {
                    "target": "lion",
                    "seed": 1,
                    "arm": "s01",
                    "progress": 0.4,
                    "calls": 4,
                },
                {
                    "target": "lion",
                    "seed": 1,
                    "arm": "pos01_neg23",
                    "progress": 0.3,
                    "calls": 4,
                },
            ]
        },
    }
    path_path = tmp_path / "path.json"
    path_path.write_text(json.dumps(path_payload), encoding="utf-8")
    path_rows, path_source = adapt_ird_path_report(
        path_payload, path=path_path, consumer_id=CONSUMER
    )

    assert {row.phase for row in kernel_rows} == {"discovery", "control"}
    assert {row.phase for row in path_rows} == {"heldout"}
    assert path_rows[0].arm_kind == "heldout-reference"
    assert path_rows[1].arm_kind == "candidate"
    assert path_rows[1].endpoint_equivalence_id == "net-zero"
    assert kernel_source["sha256"]
    assert path_source["sha256"]


def test_ird_adapter_refuses_partial_mrun_report(tmp_path: Path) -> None:
    payload = {
        "schema": "saturn-ird-p4-kernel-v1",
        "execution_receipt": {
            "mrun_job_id": "job-partial",
            "mrun_ok": False,
            "report_present": False,
        },
        "kernel": {"branches": [], "gates": {}},
    }
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HyperionError, match="successful collected"):
        adapt_ird_kernel_report(payload, path=path, consumer_id=CONSUMER)


def test_ird_depth_adapter_normalizes_and_holds_out_halves(tmp_path: Path) -> None:
    payload = {
        "schema": "saturn-ird-allmodels-p1-v1",
        "execution_receipt": {"mrun_ok": True, "report_present": True},
        "grid": {
            "tiny": {
                "cells": {
                    "identity": {
                        "n_layers": 4,
                        "necessity": {
                            "L1": {"delta_lp_base": -0.2},
                            "L3": {"delta_lp_base": -0.3},
                            "all": {"delta_lp_base": -2.0},
                            "first_half": {"delta_lp_base": -0.8},
                            "second_half": {"delta_lp_base": -1.2},
                        },
                        "gates": {"noop_max_abs": 0.0},
                    }
                }
            }
        },
    }
    path = tmp_path / "depth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows, source = adapt_ird_depth_report(
        payload,
        path=path,
        model_key="tiny",
        behavior="identity",
        consumer_id="native-final-norm-lm-head",
    )
    by_arm = {row.arm_id: row for row in rows}

    assert by_arm["all"].target_effect == 1.0
    assert by_arm["first_half"].phase == "heldout"
    assert by_arm["second_half"].phase == "heldout"
    assert by_arm["L1"].metadata["raw_target_effect"] == 0.2
    assert by_arm["noop"].phase == "control"
    assert source["adapter"] == "ird-residual-depth"

    report = compile_dynamic_circuit(
        rows,
        symbol_id="tiny.identity.depth",
        logical_symbol="identity",
        model_id="tiny",
        consumer_id="native-final-norm-lm-head",
        time_axis="transformer-layer-depth",
        address_space="residual-stream",
        residual_definition="normalized residual deletion effect",
        source_artifacts=[source],
    )
    assert report["classification"]["coordinate_kind"] == "depth"
    assert report["classification"]["primary"] == "weighted-depth-integral"
    assert "distributed-depth-coalition" in report["classification"]["properties"]
    assert report["symbol"]["execution_coordinate"]["kind"] == "depth"


def test_ar_induction_adapter_keeps_envelopes_heldout(tmp_path: Path) -> None:
    certificate_panel = {
        "metrics": {
            "clean_accuracy": 1.0,
            "source_deletion_accuracy": 0.0,
            "correct_repair_accuracy": 1.0,
        }
    }
    payload = {
        "schema": "saturn-ar-induction-circuit-certificate-report-v1",
        "status": "cross-model-ar-source-circuit-certified",
        "content_sha256": "a" * 64,
        "models": [
            {
                "model": {"name": "tiny-base", "model_id": "org/tiny-base"},
                "execution": {"consumer": "native_final_norm_lm_head"},
                "gates": {"full_universe_certified": True},
                "architecture": {"layers": 4},
                "discovery": {
                    "layer_profile": [
                        {
                            "layer": layer,
                            "necessity_effect": 0.1 * layer,
                            "repair_accuracy": 0.1 * layer,
                        }
                        for layer in range(4)
                    ],
                    "nested_envelopes": [
                        {
                            "layer_count": 2,
                            "family_id": "family-2",
                            "family": {"2": [0], "3": [0]},
                            "certificate": {"panels": [certificate_panel]},
                        }
                    ],
                },
                "confirmation": {
                    "full_universe": {
                        "certificate": {"panels": [certificate_panel]}
                    }
                },
            }
        ],
    }
    path = tmp_path / "induction.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows, source = adapt_ar_induction_report(
        payload,
        path=path,
        model_key="tiny-base",
        consumer_id="native_final_norm_lm_head",
    )
    by_arm = {row.arm_id: row for row in rows}

    assert by_arm["all"].phase == "discovery"
    assert by_arm["all"].target_effect == 1.0
    assert by_arm["envelope-2"].phase == "heldout"
    assert by_arm["envelope-2"].schedule == ((2, 1.0), (3, 1.0))
    assert source["adapter"] == "ar-induction-source-edge-depth"

    with pytest.raises(HyperionError, match="native consumer"):
        adapt_ar_induction_report(
            payload,
            path=path,
            model_key="tiny-base",
            consumer_id="wrong-consumer",
        )


def _final_dose_dominant_rows() -> list[HyperionObservation]:
    """Late-concentrated panel whose exact kernel a final dose reproduces.

    Per-cell scales stay inside the window where ``all`` clears the strong
    full-effect bar (>= 0.5) while every singleton stays subthreshold
    (< 0.5), so the coalition fires and only the final-dose share decides.
    """
    weights = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.44}
    schedules = {
        "s0": {0: 1.0},
        "s1": {1: 1.0},
        "s2": {2: 1.0},
        "s3": {3: 1.0},
        "s01": {0: 1.0, 1: 1.0},
        "s23": {2: 1.0, 3: 1.0},
        "all": {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        "all_half": {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
        "s23_double": {2: 2.0, 3: 2.0},
    }

    def effect(arm: str, scale: float) -> float:
        return sum(weights[time] * value for time, value in schedules[arm].items()) * scale

    rows: list[HyperionObservation] = []
    for cell, scale in (("lion-1", 1.1), ("wolf-2", 1.08)):
        for arm in schedules:
            rows.append(_observation(cell, arm, schedules[arm], effect(arm, scale)))
        rows.append(
            _observation(
                cell,
                "pos01_neg23",
                {0: 1.0, 1: 1.0, 2: -1.0, 3: -1.0},
                -(weights[2] + weights[3]) * scale,
                phase="heldout",
                endpoint="net-zero",
            )
        )
        rows.append(
            _observation(
                cell,
                "neg01_pos23",
                {0: -1.0, 1: -1.0, 2: 1.0, 3: 1.0},
                (weights[2] + weights[3]) * scale,
                phase="heldout",
                endpoint="net-zero",
            )
        )
        rows.append(_observation(cell, "noop", {}, 0.0, phase="control"))
    return rows


def test_final_dose_dominance_downgrades_integral_primary() -> None:
    report = _compile(_final_dose_dominant_rows())

    classification = report["classification"]
    assert classification["primary"] == "final-dose-dominant-temporal-coalition"
    assert "final-dose-dominant-response" in classification["properties"]
    assert "distributed-temporal-coalition" in classification["properties"]
    comparisons = classification["observed"]["final_dose_comparisons"]
    assert len(comparisons) == 2
    assert all(row["share"] >= 0.5 for row in comparisons)
    assert classification["thresholds"]["final_dose_share"] == 0.5
    assert verify_hyperion_report(report)["passed"] is True


def test_path_dominant_panel_keeps_integral_primary() -> None:
    report = _compile(_dynamic_rows())

    classification = report["classification"]
    assert classification["primary"] == "weighted-temporal-integral"
    assert "final-dose-dominant-response" not in classification["properties"]
    comparisons = classification["observed"]["final_dose_comparisons"]
    assert comparisons and all(row["share"] < 0.5 for row in comparisons)


def test_panel_without_final_dose_arm_skips_rival() -> None:
    rows = [row for row in _dynamic_rows() if row.arm_id != "s3"]

    report = _compile(rows)

    classification = report["classification"]
    assert classification["primary"] == "weighted-temporal-integral"
    assert "final-dose-dominant-response" not in classification["properties"]
    assert classification["observed"]["final_dose_comparisons"] == []
