"""Small offline mechanics exploration on native random Qwen and UNet models.

No language/image semantic claims. Run: python examples/explore_debugger.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from saturn_pub import (
    Act,
    Adapter,
    ExecutionPoint,
    Session,
    SlotSpec,
    SurfaceManifest,
    TransitionSpec,
)
from saturn_pub.adapters.diffusion import DiffusionAdapter
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.paths import PathSchedule, TimedAct, Trajectory, compare_trajectories
from saturn_pub.store import LocalStore
from saturn_pub.symbols import backtrace
from saturn_pub.values import describe


def qwen_probe(seed, output):
    adapter = QwenAdapter.tiny(seed, granularity="operation")
    session = adapter.session([5, 7, 11])
    session.continue_(2)
    parent = session.capture()
    payload = torch.ones_like(session.read("hidden"))

    def metrics(branch):
        state = branch.capture(retain=False).payload
        return {
            "hidden": describe(state["hidden"]),
            "logits": describe(state["logits"]),
            "keys": describe(state["keys"]),
            "token_ids": state["tokens"].tolist()[0],
        }

    native = Trajectory.run(
        session, name="native", parent=parent, steps=12, ports=("tokens",), evaluator=metrics
    )
    candidates = {}
    for name, act in (
        ("small-write", Act.add("hidden", payload, dose=1e-4)),
        ("zero-hidden", Act.zero("hidden")),
    ):
        trace = Trajectory.run(
            session,
            name=name,
            parent=parent,
            steps=12,
            ports=("tokens",),
            schedule=PathSchedule((TimedAct(0, act),)),
            evaluator=metrics,
        )
        differences = []
        for left, right in zip(native.points, trace.points):
            a, b = left.cut.payload, right.cut.payload
            differences.append(
                {
                    "offset": left.offset,
                    "hidden_different": describe(a["hidden"]) != describe(b["hidden"]),
                    "cache_different": describe(a["keys"]) != describe(b["keys"]),
                    "logits_max_abs": None
                    if a["logits"] is None
                    else float((a["logits"] - b["logits"]).abs().max()),
                    "tokens_equal": torch.equal(a["tokens"], b["tokens"]),
                }
            )
        candidates[name] = {
            "comparison": compare_trajectories(native, trace),
            "differences": differences,
            "generated": trace.points[-1].metrics["token_ids"][3:],
        }
        if name == "small-write":
            # Clear the temporary hidden slot through native token commit, then restart.
            restorable = Session.from_cut(adapter, trace.points[5].cut)
            restorable.restore(parent)
            restored = restorable.replay(parent, steps=12)
            candidates[name]["whole_cut_rewind_exact"] = describe(
                restored.capture(retain=False).payload
            ) == describe(native.points[-1].cut.payload)
            if seed == 7:
                store = LocalStore(output / "qwen-store")
                store.save(parent)
                saved = trace.points[5].cut
                store.save(saved)
                child = json.loads(
                    subprocess.check_output(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--replay-qwen",
                            str(seed),
                            str(output / "qwen-store"),
                            saved.fingerprint,
                            "7",
                        ],
                        text=True,
                    )
                )
                final = trace.points[-1].cut
                candidates[name]["fresh_process_suffix_exact"] = (
                    child["payload"] == describe(final.payload)
                    and child["cut"] == final.fingerprint
                )
                assert candidates[name]["fresh_process_suffix_exact"]
            assert candidates[name]["whole_cut_rewind_exact"]
    macro_session = adapter.session([5, 7, 11])
    macro_session.step(granularity="token")
    return {
        "seed": seed,
        "model": adapter.model_identity,
        "native_generated": native.points[-1].metrics["token_ids"][3:],
        "candidates": candidates,
        "macro_backtrace": backtrace(macro_session, "tokens"),
        "parent_unchanged": session.capture(retain=False).fingerprint == parent.fingerprint,
    }


class RepeatedRepair(Adapter):
    """Supplied integer organism with repeated repair opportunities, not learned."""

    model_identity = "supplied-repeated-repair-v1"
    execution = {"family": "repeated-repair", "program": "integer-copy-v1"}

    def boundary(self, state):
        return "halted" if state["cursor"] == 7 else f"operation:{state['cursor']}"

    def point(self, state):
        return ExecutionPoint(
            "repeated-repair", 0, "run", index=state["cursor"], next_operation=self.boundary(state)
        )

    def surface(self, state):
        return SurfaceManifest(
            (
                SlotSpec("source", role="source"),
                SlotSpec("value", role="carrier", writable=True),
                SlotSpec("output", role="consumer-output"),
                SlotSpec("cursor", role="clock"),
            ),
            "repeated-v1",
            "integer-output",
        )

    def validate(self, state):
        if (
            set(state) != {"source", "value", "output", "cursor"}
            or any(type(v) is not int for v in state.values())
            or not 0 <= state["cursor"] <= 7
        ):
            raise ValueError("invalid supplied repeated-repair closure")

    def addresses(self, state):
        return tuple(state)

    def transition(self, state):
        repair = state["cursor"] % 2 == 0
        return TransitionSpec(
            self.boundary(state),
            ("source",) if repair else ("value",),
            ("value", "output", "cursor") if repair else ("output", "cursor"),
            footprint="declared",
        )

    def advance(self, state):
        if state["cursor"] == 7:
            raise ValueError("halted")
        value = state["source"] if state["cursor"] % 2 == 0 else state["value"]
        return {**state, "value": value, "output": value, "cursor": state["cursor"] + 1}


def repeated_probe():
    session = Session(RepeatedRepair(), {"source": 7, "value": 7, "output": 7, "cursor": 0})
    parent = session.capture()
    native = Trajectory.run(
        session, name="native", parent=parent, steps=7, ports=("value", "output")
    )
    candidate = Trajectory.run(
        session,
        name="two-deletions",
        parent=parent,
        steps=7,
        ports=("value", "output"),
        schedule=PathSchedule((TimedAct(0, Act.zero("value")), TimedAct(3, Act.zero("value")))),
    )
    diff = compare_trajectories(native, candidate)
    assert diff["observable_windows"]["value"] == [
        {"diverged_at": 0, "recovered_at": 1},
        {"diverged_at": 3, "recovered_at": 5},
    ]
    return diff


def diffusion_probe():
    adapter = DiffusionAdapter.tiny(granularity="operation")
    session = adapter.session(steps=3)
    parent = session.capture()
    delta = torch.ones_like(session.read("latent")) * 0.01
    native = Trajectory.run(session, name="native", parent=parent, steps=6, ports=("latent",))
    rows = {}
    for offset in (0, 2, 4):
        trace = Trajectory.run(
            session,
            name=f"write-{offset}",
            parent=parent,
            steps=6,
            ports=("latent",),
            schedule=PathSchedule((TimedAct(offset, Act.add("latent", delta)),)),
        )
        rows[str(offset)] = {
            "comparison": compare_trajectories(native, trace),
            "final_latent_max_abs": float(
                (native.points[-1].values["latent"] - trace.points[-1].values["latent"]).abs().max()
            ),
        }
    at_scheduler = session.fork(parent)
    at_scheduler.step()
    prediction = at_scheduler.inspect().surface["slots"]
    return {
        "model": adapter.model_identity,
        "timing_arms": rows,
        "prediction_slot": next(row for row in prediction if row["name"] == "noise_prediction"),
        "prediction_readable": "noise_prediction" in at_scheduler.inspect().slots,
        "parent_unchanged": session.capture(retain=False).fingerprint == parent.fingerprint,
    }


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    output = Path("outputs/debugger-exploration")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "offline tiny random native models; mechanics exploration only",
        "qwen": [qwen_probe(seed, output) for seed in (7, 11, 19)],
        "diffusion": diffusion_probe(),
        "repeated_repairs": repeated_probe(),
    }
    report["wall_seconds"] = time.perf_counter() - start
    root = Path(__file__).resolve().parent.parent
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__).resolve(),
            *sorted((root / "src/saturn_pub").glob("*.py")),
            *sorted((root / "src/saturn_pub/adapters").glob("*.py")),
        ]
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in report["qwen"]:
        print(
            "Qwen",
            row["seed"],
            "native",
            row["native_generated"],
            "small-write",
            row["candidates"]["small-write"]["generated"],
            "zero-hidden",
            row["candidates"]["zero-hidden"]["generated"],
        )
    print(
        "Diffusion final max errors:",
        {
            offset: row["final_latent_max_abs"]
            for offset, row in report["diffusion"]["timing_arms"].items()
        },
    )
    print("Report:", output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-qwen", nargs=4, metavar=("SEED", "STORE", "CUT", "STEPS"))
    args = parser.parse_args()
    if args.replay_qwen:
        seed, directory, cut_id, steps = args.replay_qwen
        torch.set_num_threads(1)
        cut = LocalStore(directory).load(cut_id)
        session = Session.from_cut(QwenAdapter.tiny(int(seed), granularity="operation"), cut)
        for _ in range(int(steps)):
            session.step()
        final = session.capture(retain=False)
        print(json.dumps({"payload": describe(final.payload), "cut": final.fingerprint}))
    else:
        main()
