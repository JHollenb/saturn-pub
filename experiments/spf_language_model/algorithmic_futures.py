"""Common-parent, uninterrupted future panels using public Saturn transactions.

Snapshots, isolation, rollback and endpoint policy belong to TrainingTransaction.
This helper supplies orchestration and immutable custody, not another checkpoint
implementation. Negative intermediate observations never truncate a future.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from saturn_pub.training import (
    EvaluationScores,
    ImmutableDirectoryReceiptStore,
    PromotionObjective,
    PromotionPolicy,
    TrainingIntervalError,
    TrainingTransaction,
)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("future/parent names require letters, digits, dot, dash or underscore")
    return value


def _record(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)


@dataclass(frozen=True)
class FutureRequest:
    name: str
    steps: int
    train_step: Callable
    observe_at: tuple[int, ...] = (16, 32, 64)


class AlgorithmicFutures:
    """Execute candidate horizons from one exact model/optimizer/sampler parent.

    Call capture_parent, propose_futures, then commit(name) or commit(None).
    Train callbacks advance the declared cursor; physical attempts remain outside
    that cursor. Evaluators may change modes, RNG, cursor and declared diagnostic
    state; public Saturn isolates those changes and refuses parameter mutations.
    """

    def __init__(
        self,
        model,
        optimizer,
        cursor,
        root,
        identity,
        *,
        mutable=None,
        generators=None,
        policy=None,
        max_horizon=64,
        evaluation_contract=None,
    ):
        self.model, self.optimizer, self.cursor = model, optimizer, cursor
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = copy.deepcopy(dict(identity))
        self.evaluation_contract = copy.deepcopy(dict(evaluation_contract or {}))
        if self.evaluation_contract.get("selection_split") in {
            "sealed_test",
            "sealed_audit",
            "cold_audit",
            "heldout",
            "test",
        }:
            raise ValueError("held-out evaluation cannot select candidate futures")
        self.mutable, self.generators = dict(mutable or {}), dict(generators or {})
        self.policy = policy or PromotionPolicy(
            "lexicographic", (PromotionObjective("validation", "bpb", "minimize"),)
        )
        if isinstance(max_horizon, bool) or not isinstance(max_horizon, int) or max_horizon < 1:
            raise ValueError("max_horizon must be a positive bounded integer")
        self.max_horizon = max_horizon
        self.physical_attempts = 0
        self.evaluator_calls = 0
        self.parent, self.parent_scores, self.round_root = None, None, None
        self.score_binding = None
        self._parent_evaluator = None
        self.results, self._endpoints = {}, {}
        self.boundary = self._controller("boundary", self.root / "boundary-receipts")

    def _controller(self, name, receipt_root):
        if isinstance(self.cursor, dict):

            def capture():
                return copy.deepcopy(self.cursor)

            def restore(value):
                self.cursor.clear()
                self.cursor.update(copy.deepcopy(value))
        else:
            capture, restore = self.cursor.state_dict, self.cursor.load_state_dict
        return TrainingTransaction(
            controller_id=_name(name),
            model=self.model,
            optimizer=self.optimizer,
            cursor_capture=capture,
            cursor_restore=restore,
            mutable=self.mutable,
            torch_generators=self.generators,
            policy=self.policy,
            max_interval_steps=self.max_horizon,
            receipt_store=ImmutableDirectoryReceiptStore(receipt_root),
        )

    def _archive(self, snapshot, directory, name):
        snapshot.verify()
        path = directory / f"{name}-{snapshot.fingerprint}.pt"
        with path.open("xb") as stream:
            torch.save(snapshot.to_state_dict(), stream)
        return {
            "path": str(path),
            "fingerprint": snapshot.fingerprint,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def _evaluate(self, evaluator):
        self.evaluator_calls += 1
        return EvaluationScores.from_value(evaluator())

    def capture_parent(self, evaluator, *, parent_id="parent-000"):
        directory = self.root / _name(parent_id)
        directory.mkdir(exist_ok=False)
        self.parent = self.boundary.capture()
        observer = self._controller("parent-observer", directory / "parent-receipts")
        receipt = observer.establish_baseline(lambda: self._evaluate(evaluator)).to_dict()
        if self.boundary.current_state_fingerprint() != self.parent.fingerprint:
            self.boundary.restore(self.parent)
            raise RuntimeError("public baseline isolation failed to preserve common parent")
        self.parent_scores = EvaluationScores.from_value(receipt["candidate"]["evaluation"])
        self.score_binding = copy.deepcopy(
            {
                "training_closure_fingerprint": self.parent.fingerprint,
                "model_state_fingerprint": self.parent.model_state_fingerprint,
                "identity": self.identity,
                "evaluation_contract": self.evaluation_contract,
            }
        )
        self._parent_evaluator = evaluator
        self.round_root, self.results, self._endpoints = directory, {}, {}
        row = {
            "identity": self.identity,
            "evaluation_contract": self.evaluation_contract,
            "archive": self._archive(self.parent, directory, "parent"),
            "receipt": receipt,
            "score_binding": self.score_binding,
            "physical_attempts": self.physical_attempts,
            "terminal_status": "not-assessed",
        }
        _record(directory / "parent.json", row)
        return row

    def _intermediate(self, evaluator, directory, step):
        snapshot = self.boundary.capture()
        archive = self._archive(snapshot, directory, f"step-{step:03d}")
        observer = self._controller(f"observer-{step}", directory / f"step-{step:03d}-receipts")
        row = {
            "step": step,
            "archive": archive,
            "selection_authority": False,
            "identity": self.identity,
            "evaluation_contract": self.evaluation_contract,
            "terminal_status": "not-assessed",
        }
        try:
            receipt = observer.establish_baseline(lambda: self._evaluate(evaluator)).to_dict()
            row.update(
                instrument_status="ran",
                evaluation=receipt["candidate"]["evaluation"],
                receipt=receipt,
            )
        except Exception as error:
            row.update(instrument_status="error", error_type=type(error).__name__, error=str(error))
        if self.boundary.current_state_fingerprint() != snapshot.fingerprint:
            self.boundary.restore(snapshot)
            raise RuntimeError("intermediate observer failed exact isolation")
        _record(directory / f"step-{step:03d}-observation.json", row)
        return row

    def propose_futures(self, requests, evaluator):
        if self.parent is None:
            raise RuntimeError("capture a common parent before proposing futures")
        if (
            evaluator is not self._parent_evaluator
            or self.identity != self.score_binding["identity"]
            or self.evaluation_contract != self.score_binding["evaluation_contract"]
        ):
            raise ValueError(
                "common-parent scorer/input/consumer binding changed; recapture parent"
            )
        requests = tuple(requests)
        names = [_name(request.name) for request in requests]
        if len(set(names)) != len(names) or any(name in self.results for name in names):
            raise ValueError("candidate names must be unique for this parent")
        for request in requests:
            if (
                isinstance(request.steps, bool)
                or not isinstance(request.steps, int)
                or not 1 <= request.steps <= self.max_horizon
                or not callable(request.train_step)
            ):
                raise ValueError("candidate requires a bounded step count and train callback")
            if any(
                isinstance(step, bool) or not isinstance(step, int) or step < 1
                for step in request.observe_at
            ):
                raise ValueError("observation steps must be positive integers")
        for request in requests:
            directory = self.round_root / request.name
            directory.mkdir(exist_ok=False)
            self.boundary.restore(self.parent)
            if self.boundary.current_state_fingerprint() != self.parent.fingerprint:
                raise RuntimeError("candidate did not start from exact common parent")
            controller = self._controller(request.name, directory / "receipts")
            observations, partial_archives, endpoint_archive = [], [], None
            start_attempts = self.physical_attempts

            def bound_parent_scores():
                # One evaluator call per parent. This reuse is explicitly bound
                # to the captured complete closure and declared input/consumer
                # contract; it is never reused after any model/cursor drift.
                if (
                    controller.current_state_fingerprint()
                    != self.score_binding["training_closure_fingerprint"]
                ):
                    raise RuntimeError("cached parent evaluation binding changed")
                scores = self.parent_scores.to_dict()
                scores["evidence"] = {
                    **scores.get("evidence", {}),
                    "cached_parent_score_binding": self.score_binding,
                }
                return scores

            baseline = controller.establish_baseline(bound_parent_scores).to_dict()
            observation_steps = set(request.observe_at) - {request.steps}

            def train(index):
                self.physical_attempts += 1
                try:
                    metadata = request.train_step(index)
                except Exception as error:
                    archive = None
                    try:
                        archive = self._archive(
                            self.boundary.capture(), directory, f"error-step-{index + 1:03d}"
                        )
                        partial_archives.append(archive)
                    except Exception:
                        pass
                    _record(
                        directory / f"update-{index + 1:03d}-error.json",
                        {
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "partial_archive": archive,
                        },
                    )
                    raise
                _record(directory / f"update-{index + 1:03d}.json", metadata or {})
                if index + 1 in observation_steps:
                    observations.append(self._intermediate(evaluator, directory, index + 1))
                return metadata

            def endpoint():
                nonlocal endpoint_archive
                snapshot = self.boundary.capture()
                self._endpoints[request.name] = snapshot
                endpoint_archive = self._archive(snapshot, directory, "endpoint")
                scores = self._evaluate(evaluator)
                row = {
                    "step": request.steps,
                    "archive": endpoint_archive,
                    "selection_authority": True,
                    "evaluation": scores.to_dict(),
                    "identity": self.identity,
                    "evaluation_contract": self.evaluation_contract,
                    "instrument_status": "ran",
                    "terminal_status": "not-assessed",
                }
                _record(directory / "endpoint-observation.json", row)
                observations.append(row)
                return scores

            try:
                receipt = controller.run_interval(
                    interval_id="future", steps=request.steps, train_step=train, evaluator=endpoint
                ).to_dict()
                status = "complete"
            except TrainingIntervalError as error:
                receipt, status = error.receipt.to_dict(), "error"
                if endpoint_archive is None:
                    try:
                        # The public controller has already restored parent; its
                        # retained error receipt identifies the rejected future.
                        _record(
                            directory / "incomplete-closure.json",
                            {"status": "no-executable-endpoint", "receipt": receipt},
                        )
                    except FileExistsError:
                        pass
            finally:
                self.boundary.restore(self.parent)
            row = {
                "name": request.name,
                "steps": request.steps,
                "status": status,
                "common_parent_fingerprint": self.parent.fingerprint,
                "post_panel_fingerprint": self.boundary.current_state_fingerprint(),
                "baseline": baseline,
                "observations": observations,
                "partial_archives": partial_archives,
                "endpoint_archive": endpoint_archive,
                "receipt": receipt,
                "physical_attempts": self.physical_attempts - start_attempts,
                "negative_intermediate_stopping": False,
                "terminal_status": "not-assessed",
            }
            _record(directory / "future.json", row)
            self.results[request.name] = row
        return copy.deepcopy(self.results)

    def best_future(self):
        """Rank endpoints with the declared policy, starting from the parent."""
        if self.parent_scores is None:
            raise RuntimeError("no evaluated common parent")
        best_name, best_scores = None, self.parent_scores
        for name, row in self.results.items():
            if row["status"] != "complete":
                continue
            scores = EvaluationScores.from_value(row["receipt"]["candidate"]["evaluation"])
            if self.policy.compare(best_scores, scores)["promoted"]:
                best_name, best_scores = name, scores
        return best_name

    def commit_best(self):
        return self.commit(self.best_future())

    def commit(self, name=None):
        if self.parent is None:
            raise RuntimeError("no captured parent")
        if self.boundary.current_state_fingerprint() != self.parent.fingerprint:
            self.boundary.restore(self.parent)
            raise RuntimeError("live model drifted after common-parent future panel")
        selected, comparison = self.parent, None
        if name is not None:
            row = self.results[name]
            if row["status"] == "complete":
                scores = EvaluationScores.from_value(row["receipt"]["candidate"]["evaluation"])
                comparison = self.policy.compare(self.parent_scores, scores)
                if comparison["promoted"]:
                    selected = self._endpoints[name]
        self.boundary.restore(selected)
        row = {
            "selected_future": name,
            "continued_future": name if selected is not self.parent else None,
            "comparison": comparison,
            "parent_fingerprint": self.parent.fingerprint,
            "post_state_fingerprint": self.boundary.current_state_fingerprint(),
            "verified_exact": self.boundary.current_state_fingerprint() == selected.fingerprint,
            "physical_attempts": self.physical_attempts,
            "evaluator_calls": self.evaluator_calls,
            "terminal_status": "not-assessed",
        }
        _record(self.round_root / "commit.json", row)
        return row
