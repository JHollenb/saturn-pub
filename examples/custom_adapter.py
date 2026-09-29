"""A minimal JSON-only custom machine. No torch installation is required."""

from saturn_pub import Act, Adapter, Session


class Counter(Adapter):
    model_identity = "counter-v1"
    execution = {"family": "custom", "parity": "exact-integer"}

    def boundary(self, state):
        return "counter-step"

    def validate(self, state):
        if type(state.get("value")) is not int:
            raise ValueError("integer value required")

    def addresses(self, state):
        return ("value",)

    def advance(self, state):
        return {"value": state["value"] + 1}


session = Session(Counter(), {"value": 0})
parent = session.capture()
candidate = session.fork(parent)
candidate.apply(Act.add("value", 10))
candidate.continue_()
session.commit(candidate)
assert session.read("value") == 11
session.restore(parent)
assert session.read("value") == 0
print("Custom adapter transaction and exact restore verified.")
