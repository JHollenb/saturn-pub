"""Bounded navigation of a public Saturn Session using its native receipts.

The session must be owned by this navigator while in use. Reading ``point``
does not inspect tensors, capture a cut, or execute the model. Native Session
integrity checks and captures remain in force inside every operation.
"""
from __future__ import annotations

from copy import deepcopy


class NativeCutNavigator:
    def __init__(self, session, *, max_steps, on_receipt=None):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.session = session
        self.max_steps = max_steps
        self.steps = 0
        self.on_receipt = on_receipt
        # One initial capture establishes the cursor; subsequent reads are free.
        self._point = dict(session.capture(retain=False).execution_point or {})

    @property
    def point(self):
        return deepcopy(self._point)

    def _accept(self, receipt):
        self._point = deepcopy(receipt.to_dict()["execution_point"])
        if self.on_receipt is not None:
            self.on_receipt(receipt)
        return receipt

    def step(self, n=1):
        if type(n) is not int or n < 1 or self.steps + n > self.max_steps:
            raise ValueError("native navigation step budget exceeded or invalid")
        # Reserve attempts too: a failed operation must not permit an unbounded
        # retry loop. The native Session owns atomic failure/rollback behavior.
        self.steps += n
        return self._accept(self.session.continue_(n))

    def until(self, **selector):
        if not selector or not set(selector) <= set(self._point):
            raise ValueError("selector must name known execution-point fields")
        while any(self._point[key] != value for key, value in selector.items()):
            self.step()
        return self.point

    def apply(self, act):
        return self._accept(self.session.apply(act))

    def capture(self):
        cut = self.session.capture()
        self._point = dict(cut.execution_point or {})
        return cut

    def fork(self, cut):
        return type(self)(self.session.fork(cut), max_steps=self.max_steps - self.steps,
                          on_receipt=self.on_receipt)
