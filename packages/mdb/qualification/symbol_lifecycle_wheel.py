"""Qualify installed symbol navigation with model and private imports blocked."""

import argparse
import importlib.abc
import json
import sys
from pathlib import Path


class BlockRuntime(importlib.abc.MetaPathFinder):
    blocked = {"torch", "transformers", "mrun", "saturn", "common", "common_harness"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError("framework/private import during offline qualification: " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockRuntime())
    import saturn_pub
    from saturn_pub import Session, StateAddress
    from saturn_pub.contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
    from saturn_pub.core import Adapter
    from saturn_pub.symbols import SymbolBinding

    import mdb
    from mdb.project import Project
    from mdb.symbol_lifecycle import ClockWitness, SymbolLifecycle, SymbolOccurrence, SymbolUse

    for module in (mdb, saturn_pub):
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError("qualification did not import installed wheels")

    class Counter(Adapter):
        model_identity = "wheel-counter-v1"
        execution = {"family": "counter", "backend": "python", "adapter": "wheel-counter-v1"}

        def boundary(self, state):
            return "tick:" + str(state["tick"])

        def point(self, state):
            return ExecutionPoint("counter", state["tick"], "tick", next_operation="increment")

        def surface(self, state):
            return SurfaceManifest((SlotSpec("value"), SlotSpec("tick", role="clock")),
                                   "wheel-counter-state-v1", "wheel-counter-consumer")

        def validate(self, state):
            if set(state) != {"value", "tick"} or any(type(v) is not int for v in state.values()):
                raise ValueError("counter state changed")

        def addresses(self, state):
            return tuple(state)

        def transition(self, state):
            return TransitionSpec("increment", ("value", "tick"), ("value", "tick"),
                                  consumer="wheel-counter-consumer", footprint="declared")

        def advance(self, state):
            return {name: value + 1 for name, value in state.items()}

    session = Session(Counter(), {"value": 0, "tick": 0})
    parent = session.capture()
    binding = SymbolBinding.bind(session, "counter.value", (StateAddress("value"),),
        context="wheel-fixture", context_slots=(), evidence=("cas://sha256/" + parent.fingerprint,),
        claim="Supplied native counter field; mechanics fixture only.")
    receipt = session.continue_(2)
    result = session.capture()
    project = Project(args.root)
    project.save_session("wheel", session, (parent, result),
        debug_info={"symbols": [binding.to_dict()], "symbol_context": binding.context})
    clock = ClockWitness("wheel-counter-cursor", "execution-point", "wheel/counter",
        dict(parent.execution_point), provenance=("cas://sha256/" + parent.fingerprint,),
        status="observed", metadata={"source_cut": parent.fingerprint})
    definition = SymbolOccurrence.for_binding(binding, occurrence_id="counter-source",
        session_id="wheel", source_cut=parent.fingerprint, locations=binding.locations,
        branch="main", execution=binding.execution, clock=clock,
        evidence_refs=("cas://sha256/" + parent.fingerprint,))
    use = SymbolUse("counter-use", binding.to_dict()["fingerprint"], "wheel",
        "wheel-counter-consumer", source_cut=parent.fingerprint, parent_cut=parent.fingerprint,
        result_cut=result.fingerprint, receipt=receipt.fingerprint, execution=binding.execution,
        signed_observations=({"observable": "counter-change", "value": 2, "n": 1},),
        evidence_refs=("cas://sha256/" + receipt.fingerprint,))
    lifecycle = SymbolLifecycle.define(binding, definitions=(definition,), uses=(use,))
    saved = project.save_symbol_lifecycle("wheel", lifecycle)
    reopened = Project(args.root)
    if reopened.symbol_lifecycle("wheel", lifecycle.fingerprint) != saved:
        raise RuntimeError("installed lifecycle reopen differs")
    reverse = reopened.symbols_at("wheel", "value", cut_id=parent.fingerprint)
    if len(reverse["matches"]) != 1:
        raise RuntimeError("installed reverse lookup lost the definition")
    if reopened.session("wheel")["selected"] != result.fingerprint:
        raise RuntimeError("offline navigation changed native selection")
    loaded = sorted({name.split(".")[0] for name in sys.modules} & BlockRuntime.blocked)
    if loaded:
        raise RuntimeError("blocked runtime modules loaded")
    report = {"schema": "mdb-symbol-lifecycle-installed-qualification-v1",
              "mechanics_status": "valid", "instrument_status": "valid",
              "terminal_status": "not-assessed", "lifecycle_id": lifecycle.fingerprint,
              "binding_fingerprint": binding.to_dict()["fingerprint"],
              "parent_cut": parent.fingerprint, "result_cut": result.fingerprint,
              "reverse_matches": len(reverse["matches"]), "native_payload_reads": 0,
              "frameworks_loaded": loaded,
              "mdb": str(Path(mdb.__file__).resolve()),
              "saturn_pub": str(Path(saturn_pub.__file__).resolve())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
