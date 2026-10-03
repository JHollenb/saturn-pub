# Induction certificate: preregister, measure, and sign a bounded circuit claim

A live attention site is a candidate, not a circuit (trap 7 in the [Instrument
Trial](trial.md)). `saturn_pub.certificate` turns a candidate direct-source attention
family into a *fail-closed, preregistered* certificate: a researcher freezes the circuit
family, the control arms, the held-out panels, and the numeric thresholds **in advance**;
a worker then supplies the measured per-example outcomes; and the gate returns a signed
(content-hashed) certificate, or -- the instant a gate fails -- the same object with the
failing gate named. Nothing is tuned after the outcomes arrive.

The workload is arbitrary-token relational induction:

```text
t0 t1 t2 t3 t4 t5  t0 t1 t2 -> t3
```

The tokens and the cue position change on every example and every panel, so the claimed
role is not a phrase or a token ID -- it is the operation "retrieve the token that
followed the earlier matching cue."

The module has three layers. Importing the package never imports torch.

## 1. The gate (stdlib only)

```python
from saturn_pub.certificate import InductionCircuitPolicy, certify_induction_route

policy = InductionCircuitPolicy()  # frozen thresholds + native-consumer contract
certificate = certify_induction_route(panels, policy)
assert certificate["certified"] is True  # or read certificate["panels"][i]["gates"]
```

Each panel supplies five aligned per-example branches -- `clean`, `source_deletion`,
`match_deletion`, `correct_repair`, `wrong_repair` -- as `{correct: [...], margin: [...]}`
plus `classes` and an `execution` block naming its consumer. The gate computes seven
gates per panel and two over the set:

| Gate | What it requires |
| --- | --- |
| `clean_behavior` | clean accuracy at least `clean_min_accuracy` |
| `source_necessity` | Wilson upper bound of the source-cut accuracy at or below chance + slack |
| `position_specificity` | the neighbor-cue cut barely dents accuracy, and stays high |
| `correct_repair_sufficiency` | restoring the family's clean output recovers clean accuracy |
| `wrong_donor_specificity` | a same-shaped different-answer donor stays at chance, far below repair |
| `repair_replay_fidelity` | repair reproduces the clean decision and candidate margin |
| `native_consumer_continuation` | every branch terminated at the declared native consumer |
| `replication` (set) | at least `min_panels` distinct panels |
| `all_panels_certified` (set) | every panel passed every gate |

`source_necessity` and `wrong_donor_specificity` use a Wilson score upper bound, so a few
lucky successes on a small panel do not pass a necessity or specificity gate.

The `native_consumer_continuation` gate is a declarable `ConsumerContract`: by default it
only requires the branch consumer to equal `native_final_norm_lm_head`, so the same
policy certifies a CPU specimen and a GPU run. Pin the substrate when a certificate must
name it:

```python
from saturn_pub.certificate import ConsumerContract

policy = InductionCircuitPolicy(
    consumer=ConsumerContract(backends=("native-decoder-v1",), device_prefixes=("cuda",)),
)
```

## 2. Measure the panel on a native decoder adapter

```python
from saturn_pub.adapters import load
from saturn_pub.certificate import InductionPanelSpec
from saturn_pub.certificate.panel import certify_decoder_induction

adapter = load("Qwen/Qwen2.5-0.5B", dtype="float32")  # eager, frozen, native consumer
specs = [
    InductionPanelSpec("len12-seed-a", seed=20260812, length=12),
    InductionPanelSpec("len16-seed-b", seed=20260813, length=16),
]
bundle = certify_decoder_induction(adapter, specs, InductionCircuitPolicy())
print(bundle["certificate"]["certified"], bundle["certificate"]["content_sha256"])
```

`measure_induction_panel` runs the five-branch battery over the complete direct-source
parent (every attention layer and query head). Every branch terminates at the adapter's
own final norm and output embedding (`model(...).logits`), so the native-consumer gate is
about the real model, never a surrogate readout. The attention-edge deletion and the
pre-`o_proj` repair are finer than the decoder adapter's declared Session state ports, so
they are applied as explicit forward-hook contexts on the adapter's resident native model
-- the same frozen model the Session executes -- and a `Session`/`fork` consumer
cross-check (recorded in `execution.session_consumer_crosscheck`) confirms the stepped
native suffix reproduces the batched clean decision. Supported attention families are the
llama-style decoders with a contiguous per-head `self_attn.o_proj`: `llama`, `mistral`,
`mixtral`, `gemma`, `qwen2`, `qwen3`; other registered decoder families are refused
fail-closed.

## 3. Register the certificate as evidence

A certificate is receipt-shaped (top-level `gates` + `content_sha256`), so it binds to
the [evidence plane](evidence.md) directly:

```python
from saturn_pub.evidence import ClaimsRegistry
from saturn_pub.certificate import register_certificate_claim, certificate_receipt

receipt = certificate_receipt(certificate)  # a core.Receipt seal
registry = ClaimsRegistry("claims.jsonl")
register_certificate_claim(registry, certificate, "induction-cert", "...", "certificate.json")
```

The claim status defaults to `measured` on a pass and `refuted` on a fail -- a failing
certificate is a legitimate recorded observation. If the certificate JSON on disk later
changes, `saturn_pub.evidence.verify` marks the claim stale.

## Claim boundary

The certificate is bounded to the recorded model bytes, the declared route family, the
sealed prompt panels, the native lexical consumer, and the thresholds. It is **not** a
claim that the circuit is globally minimal, unique, natural-language-general, or
architecture-universal, and the certified full direct-source parent is a distributed
family of edges, not a compact circuit. The compressed-envelope search that the owner's
private worker ran is out of scope here; this module certifies the full parent. A failing
gate is reported, never tuned away.

## Runnable example

```bash
python examples/induction_certificate.py
```

It freezes a policy and two panels, measures them on a tiny random Qwen2 specimen, and
reads the gates. Random weights do not perform induction, so `clean_behavior` fails and
the certificate is honestly NOT certified -- while the mechanical `repair_replay_fidelity`
identity still holds exactly. The real-weight certificate on cached small decoders is in
[`experiments/induction_certificate`](../experiments/induction_certificate/README.md).
