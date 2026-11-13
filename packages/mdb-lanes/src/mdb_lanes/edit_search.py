"""Framework-free, evidence-bound finite intervention search.

The consumer supplies verdicts. This ledger never infers semantics from a tensor
distance, assumes monotonicity, or reuses evidence across different bindings.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    body = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x") as stream:
        stream.write(body + "\n")


class EditSearch:
    """Immutable plan and per-candidate freezes/outcomes; one writer per root."""

    def __init__(self, root):
        self.root = Path(root)
        self.plan = json.loads((self.root / "plan.json").read_text())
        if self.plan["schema"] != "mdb-edit-search-v1" or self.plan["identity"] != digest({k: v for k, v in self.plan.items() if k != "identity"}):
            raise ValueError("search plan identity differs")

    @classmethod
    def create(cls, root, *, binding, atoms, observable):
        """Binding must name the complete parent, model, execution and consumer."""
        required = {"parent", "model", "execution", "consumer"}
        if not required <= set(binding) or any(not binding[k] for k in required):
            raise ValueError("complete parent/model/execution/consumer binding required")
        if not 1 <= len(atoms) <= 12 or len(set(atoms)) != len(atoms) or not all(isinstance(a, str) and a for a in atoms):
            raise ValueError("one to twelve unique named atoms required")
        if not observable:
            raise ValueError("bounded observable required")
        plan = dict(schema="mdb-edit-search-v1", binding=binding, atoms=list(atoms), observable=observable)
        plan["identity"] = digest(plan)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        (root / "candidates").mkdir()
        _write(root / "plan.json", plan)
        return cls(root)

    def key(self, atoms):
        if len(set(atoms)) != len(atoms) or not set(atoms) <= set(self.plan["atoms"]):
            raise ValueError("unknown or duplicate intervention atom")
        selected = [a for a in self.plan["atoms"] if a in atoms]
        return digest(dict(plan=self.plan["identity"], atoms=selected)), selected

    def freeze(self, atoms, prediction):
        key, selected = self.key(atoms)
        value = dict(candidate=key, atoms=selected, plan=self.plan["identity"], prediction=prediction)
        value["identity"] = digest(value)
        path = self.root / "candidates" / key
        path.mkdir(exist_ok=False)
        _write(path / "freeze.json", value)
        return value

    def observe(self, candidate, *, evidence, verdict, measurements=None):
        """Verdict is sufficient/insufficient/uncertain for the declared observable.

        Evidence must include the native replay/custody audit and consumer result;
        an uncertain instrument remains uncertain rather than pruning the lattice.
        """
        if verdict not in {"sufficient", "insufficient", "uncertain"}:
            raise ValueError("unknown consumer verdict")
        path = self.root / "candidates" / candidate
        frozen = json.loads((path / "freeze.json").read_text())
        if frozen["candidate"] != candidate or frozen["identity"] != digest({k: v for k, v in frozen.items() if k != "identity"}) or self.key(frozen["atoms"])[0] != candidate:
            raise ValueError("candidate freeze differs")
        refs = [dict(path=str(Path(p).resolve(strict=True)), sha256=file_digest(p)) for p in evidence]
        if not refs:
            raise ValueError("native evidence references required")
        value = dict(candidate=candidate, freeze_sha256=file_digest(path / "freeze.json"), evidence=refs,
                     verdict=verdict, measurements=measurements or {})
        _write(path / "outcome.json", value)
        return value

    def status(self):
        records = []
        for path in sorted((self.root / "candidates").iterdir()):
            frozen = json.loads((path / "freeze.json").read_text())
            if frozen["plan"] != self.plan["identity"] or frozen["candidate"] != path.name or frozen["identity"] != digest({k: v for k, v in frozen.items() if k != "identity"}) or self.key(frozen["atoms"])[0] != path.name:
                raise ValueError("candidate freeze differs")
            record = dict(candidate=path.name, atoms=frozen["atoms"], prediction=frozen["prediction"], verdict="pending")
            if (path / "outcome.json").exists():
                result = json.loads((path / "outcome.json").read_text())
                if result["candidate"] != path.name or result["freeze_sha256"] != file_digest(path / "freeze.json"):
                    raise ValueError("outcome binding differs")
                if result["verdict"] not in {"sufficient", "insufficient", "uncertain"} or not result["evidence"]:
                    raise ValueError("invalid consumer assessment")
                for ref in result["evidence"]:
                    if file_digest(ref["path"]) != ref["sha256"]:
                        raise ValueError("native evidence changed")
                record.update(verdict=result["verdict"], measurements=result["measurements"])
            records.append(record)
        sufficient = [r for r in records if r["verdict"] == "sufficient"]
        best_size = min((len(r["atoms"]) for r in sufficient), default=None)
        best = [r for r in sufficient if len(r["atoms"]) == best_size]
        observed = {self.key(r["atoms"])[0]: r["verdict"] for r in records}
        unresolved = []
        for size in range(best_size if best_size is not None else len(self.plan["atoms"]) + 1):
            for subset in itertools.combinations(self.plan["atoms"], size):
                if observed.get(self.key(subset)[0]) not in {"sufficient", "insufficient"}:
                    unresolved.append(list(subset))
        return dict(plan=self.plan["identity"], records=records, smallest_observed=best,
                    unresolved_smaller=unresolved, minimum_cardinality_established=bool(best) and not unresolved,
                    scope="Only the declared finite atoms, parent, execution and consumer observable; no monotonicity assumed.")

    def propose(self, limit=8):
        if not 1 <= limit <= 4096:
            raise ValueError("positive bounded proposal budget required")
        status = self.status()
        seen = {r["candidate"] for r in status["records"]}
        # Examine small remaining subsets, including combinations outside any
        # successful branch. Failed supersets never eliminate their subsets.
        return [dict(atoms=s, candidate=self.key(s)[0]) for s in status["unresolved_smaller"]
                if self.key(s)[0] not in seen][:limit]

    def summary(self):
        status = self.status()
        return dict(plan=status["plan"], frozen=len(status["records"]),
                    observed=sum(r["verdict"] != "pending" for r in status["records"]),
                    uncertain=sum(r["verdict"] == "uncertain" for r in status["records"]),
                    smallest_observed=[dict(candidate=r["candidate"], atoms=r["atoms"]) for r in status["smallest_observed"]],
                    unresolved_smaller=status["unresolved_smaller"],
                    minimum_cardinality_established=status["minimum_cardinality_established"], scope=status["scope"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("init"); create.add_argument("root", type=Path); create.add_argument("--plan", type=Path, required=True)
    for name in ("status", "propose", "freeze", "observe"):
        p = sub.add_parser(name); p.add_argument("root", type=Path)
        if name == "status": p.add_argument("--summary", action="store_true")
        if name == "propose": p.add_argument("--limit", type=int, default=8)
        if name == "freeze": p.add_argument("--atoms", nargs="*", default=[]); p.add_argument("--prediction", type=Path, required=True)
        if name == "observe":
            p.add_argument("candidate"); p.add_argument("--evidence", type=Path, nargs="+", required=True)
            p.add_argument("--verdict", choices=["sufficient", "insufficient", "uncertain"], required=True)
            p.add_argument("--measurements", type=Path)
    args = parser.parse_args(argv)
    if args.command == "init":
        search = EditSearch.create(args.root, **json.loads(args.plan.read_text())); result = search.status()
    else:
        search = EditSearch(args.root)
        if args.command == "status": result = search.summary() if args.summary else search.status()
        elif args.command == "propose": result = search.propose(args.limit)
        elif args.command == "freeze": result = search.freeze(args.atoms, json.loads(args.prediction.read_text()))
        else: result = search.observe(args.candidate, evidence=args.evidence, verdict=args.verdict,
                                      measurements=json.loads(args.measurements.read_text()) if args.measurements else None)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
