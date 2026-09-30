"""Readable, recorded debugger walkthrough with fresh-process native/candidate replay.

Run from the checkout: python examples/debugger_replay.py
No model download; seed-7 random Qwen2, CPU, two decoder layers, two generated tokens.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.cli import Debugger
from saturn_pub.values import describe


def observation(debugger):
    session = debugger.session
    payload = session.capture(retain=False).payload
    hidden = payload["hidden"]
    logits = payload["logits"]
    return {
        "boundary": session.inspect().boundary,
        "tokens": session.read("tokens").tolist()[0],
        "hidden_rms": None if hidden is None else hidden.square().mean().sqrt().item(),
        "kv_lengths": [key.shape[-2] for key in payload["keys"]],
        "top_token": None if logits is None else logits.argmax(-1).item(),
        "payload": describe(payload),
    }


def replay(directory, identifier):
    debugger = Debugger(QwenAdapter.tiny(seed=7).session([5, 7, 11]))
    debugger.execute(f"load {directory} {identifier} parent")
    debugger.execute("restore parent")
    results = {}
    for branch in ("native", "candidate"):
        debugger.execute(f"fork {branch} parent")
        debugger.execute(f"use {branch}")
        if branch == "candidate":
            debugger.execute("zero hidden")
        debugger.execute("continue 6")
        results[branch] = observation(debugger)
        debugger.execute("use root")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/debugger-replay")
    parser.add_argument("--replay", nargs=2, metavar=("DIRECTORY", "DIGEST"))
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(*args.replay)))
        return

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    debugger = Debugger(QwenAdapter.tiny(seed=7).session([5, 7, 11]))
    commands, transcript = [], []

    def command(line):
        prompt = debugger.active
        result = debugger.execute(line)
        commands.append({"command": line, "result": result})
        transcript.append(f"{prompt}> {line}")
        if line == "inspect":
            frame = observation(debugger)
            transcript.append(
                f"boundary={frame['boundary']} tokens={frame['tokens']} "
                f"KV lengths={frame['kv_lengths']} hidden RMS={frame['hidden_rms']}"
            )
        elif line.startswith("read"):
            transcript.append(json.dumps(result))
        elif isinstance(result, dict) and "operation" in result:
            transcript.append(
                f"{result['operation']}: "
                + ("exact restore verified" if result.get("verified_exact") else "recorded")
            )
        elif line.startswith("compare"):
            transcript.append(f"equal_payload={result['equal_payload']}")
        else:
            transcript.append(json.dumps(result))
        return result

    command("inspect")
    command("step")  # embedding
    command("inspect")
    command("step")  # decoder layer 0; layer cursor now 1
    command("inspect")
    command("capture parent")
    identifier = command(f"save {directory} parent")["saved"]

    results, traces = {}, {}
    for branch in ("native", "candidate"):
        command(f"fork {branch} parent")
        command(f"use {branch}")
        if branch == "candidate":
            command("zero hidden")
            command("inspect")
        traces[branch] = [observation(debugger)]
        for _ in range(6):  # remaining 2 steps + a full 4-step token
            command("step")
            traces[branch].append(observation(debugger))
        command("read tokens")
        results[branch] = observation(debugger)
        command("use root")

    command("use native")
    comparison = command("compare candidate")
    command("use root")
    parent_unchanged = debugger.session.capture(retain=False).fingerprint == identifier
    command("commit candidate")
    command("read tokens")
    restoration = command("restore parent")
    command("inspect")

    # Rebuild the identical model and load the durable tensor closure in another interpreter.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--replay", str(directory), identifier],
        capture_output=True,
        text=True,
        check=True,
    )
    fresh = json.loads(completed.stdout)
    exact_replay = {branch: fresh[branch] == results[branch] for branch in results}
    if not parent_unchanged or not restoration["verified_exact"] or not all(exact_replay.values()):
        raise RuntimeError("debugger integrity/replay check did not hold")
    report = {
        "specimen": "random Qwen2 seed=7, CPU float32, 2 layers, vocab=64",
        "generation_budget": 2,
        "parent": identifier,
        "parent_unchanged_before_commit": parent_unchanged,
        "branches": results,
        "traces": traces,
        "comparison": comparison,
        "restore": restoration,
        "fresh_process_exact_replay": exact_replay,
        "commands": commands,
        "limitations": "Mechanics demonstration; token IDs carry no trained language meaning.",
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for step, (native, candidate) in enumerate(zip(traces["native"], traces["candidate"])):
        rows.append(
            f"| {step} | {native['boundary']} | {native['tokens']} | "
            f"{candidate['tokens']} | {native['kv_lengths']} |"
        )
    walkthrough = (
        "# Debugging and durable replay\n\n"
        "Random native Qwen2, seed 7, CPU float32, two layers. Token IDs demonstrate execution "
        "control, not language capability. `layer:1` means layer 0 has finished; layer 1 is next.\n\n"
        "Stop after layer 0, inspect the carrier and KV cursor, save the full state, and fork. "
        "Zero only the candidate hidden carrier, then step both branches through the unchanged "
        "native layer/readout. Each branch emits two tokens.\n\n"
        "| Suffix steps | Boundary | Native tokens | Candidate tokens | KV lengths, both |\n"
        "| --- | --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + "\n\nThe parent stays unchanged while branches run. Commit adopts the candidate; restore "
        "returns exactly to the saved parent, including both layers' K/V. A fresh Python process "
        "reconstructs the same model, verifies and loads the saved cut, and replays both "
        "futures. All declared payloads (including logits and K/V) match exactly.\n\n"
        + f"Fresh-process exact replay: `{exact_replay}`.\n\n"
        + "## Recorded debugger commands\n\n```text\n"
        + "\n".join(transcript)
        + "\n```\n"
    )
    (directory / "walkthrough.md").write_text(walkthrough)
    print(walkthrough)


if __name__ == "__main__":
    main()
