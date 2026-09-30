# Reversible training

TrainingTransaction adapts Saturn's deterministic checkpoint controller. The unit is a short
candidate update interval, followed by isolated downstream evaluation and a continuation decision.

```text
capture accepted closure → bounded update → inspect component and native future
                        → retain evidence → promote or restore exactly
```

The controller captures parameters, persistent and nonpersistent buffers, optimizer state,
optimizer defaults and parameter topology, gradients, module modes, requires_grad flags,
Python/NumPy/Torch RNG, named Torch generators, and caller-owned cursor. `mutable` callbacks
register additional caches or hook state. A cursor must include sampler/dataloader position;
an arbitrary iterator is not automatically rewindable.

```python
from saturn_pub.training import TrainingTransaction, PromotionObjective, PromotionPolicy

controller = TrainingTransaction(
    controller_id="my-writer",
    model=writer,
    optimizer=optimizer,
    policy=PromotionPolicy(
        "lexicographic", (PromotionObjective("validation", "loss", "minimize"),)
    ),
    cursor_capture=capture_cursor,
    cursor_restore=restore_cursor,
    mutable={"cache": (capture_cache, restore_cache)},
    max_interval_steps=4,
)
controller.establish_baseline(evaluate)
receipt = controller.run_interval(
    interval_id="candidate-1",
    steps=2,
    train_step=train_step,
    evaluator=evaluate,
)
```

The evaluator returns EvaluationScores with nonempty `validation`, `autonomous_rollout`, and
`trace_endpoints` maps. Keep all three views visible. Policy objectives select only named metrics;
they do not require every measurement to improve. Lexicographic and Pareto continuation policies
are supported. Tolerances are operational guardbands, not scientific calibration.

`PromotionPolicy.require_all_score_sources` controls which score sources must appear
in the policy's objectives; it does not enable checkpointing or rollback. Its default
is false so an objective can select one relevant metric while EvaluationScores still
retains all three measurement views. Set it true only when your continuation policy
explicitly requires objectives from all three sources.

Use frozen recipient continuation to evaluate a trained writer where possible. Trace imitation
can initialize a candidate; native behavior decides whether it is useful. The offline example
trains a small hidden-state writer through frozen Qwen readout, measures generated continuation,
then deliberately proposes a harmful update and verifies rollback.

Evaluators run with RNG/cursor/mode isolation. Model, optimizer, or gradient mutations are refused
and the candidate interval is restored. Exceptions during training also produce a retained error
receipt and restore the accepted state. Receipts preserve rejected scores and fingerprints; rejected
parameter payloads are not automatically archived. Explicitly capture/archive a candidate if it
must remain executable afterward. The controller holds the accepted training closure in memory;
LocalStore stores execution StateCuts, not arbitrary optimizer state dictionaries.

Define update data, feedback-monitor data, and final held-out data separately. Never select an
interval using the final held-out result. In the offline example the desired token and feedback
context are supplied; two fresh context probes run only after training decisions. Those small
probes are mechanics illustrations, not evidence of general capability learning.

Never mutate a resident inference adapter's model in place. Train a separate component or create
a fresh adapter after changing weights so model identity and state compatibility remain correct.
External API calls, files written by user callbacks, and undeclared Python state are outside rollback.

## Parameter-free structural state

`TrainingStateBoundary` also accepts `optimizer=None` when the supplied `torch.nn.Module` has no
parameters. This supports exact snapshots of caller-owned structural state—such as a graph, rules,
memory, and cursor—through `cursor_capture` and `cursor_restore`. The snapshot payload can be
rehydrated with `TrainingStateSnapshot.from_state_dict` and restored after graph edits in a fresh
runtime. Any nonempty optimizer state is refused for an optimizer-free snapshot.

Run `python examples/structural_rewind.py` for a parent/candidate/cold-payload round trip. This
establishes structural custody and rollback mechanics. It does not establish learned growth,
functional preservation after mutation, or usefulness of a structural policy.
