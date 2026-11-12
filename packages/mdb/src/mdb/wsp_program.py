"""Small executable semantic requests bound to an evidence-backed account.

This prototype lowers an authored digit-rewrite request through supplied,
checkpoint-qualified operand/read locations. It does not infer those roles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .wsp_weights import digest


@dataclass(frozen=True)
class DigitRewrite:
    operand: str
    column: int
    source: int
    target: int

    def __post_init__(self):
        if (self.operand not in {"a", "b"} or type(self.column) is not int or self.column < 0
                or any(type(x) is not int or not 0 <= x <= 9 for x in (self.source, self.target))
                or self.source == self.target):
            raise ValueError("distinct digit values and a supported operand/column required")

    @property
    def symbol(self):
        return "operand." + self.operand + ".column" + str(self.column)

    def apply(self, a, b):
        if any(type(x) is not int or x < 0 for x in (a, b)):
            raise ValueError("nonnegative integer operands required")
        values = {"a": a, "b": b}
        value = values[self.operand]
        if value // (10 ** self.column) % 10 == self.source:
            values[self.operand] += (self.target - self.source) * 10 ** self.column
        return values["a"], values["b"]

    def lower(self, account, *, checkpoint_sha256, execution):
        """Resolve candidate writers from a supplied account, not a layer scan.

        Qualified bindings and source hashes protect identity. Historical role
        evidence remains a hypothesis on this newly declared context.
        """
        if (account.get("checkpoint_sha256") != checkpoint_sha256
                or account.get("execution") != execution or not account.get("evidence")):
            raise ValueError("computation account identity or evidence missing")
        role = account.get("symbols", {}).get(self.symbol)
        if not role or not role.get("writers") or not role.get("consumer"):
            raise ValueError("symbol has no supported source-to-consumer account")
        if any(not record.get("sha256") or not record.get("path") for record in account["evidence"]):
            raise ValueError("evidence content identities required")
        body = {"schema": "mdb-computation-edit-recipe-v1", "program": asdict(self),
            "symbol": self.symbol, "account_sha256": digest(account),
            "writers": role["writers"], "consumer": role["consumer"],
            "predictions": {"matched": "sum of rewritten operands", "unmatched": "original sum",
                "wrong_location": "may disrupt; should not systematically implement the same rewrite",
                "lowering": "permanent weights may affect additional contexts; measure separately"},
            "status": "experimental request; native intervention and weight realization unmeasured"}
        return {**body, "identity": digest(body)}
