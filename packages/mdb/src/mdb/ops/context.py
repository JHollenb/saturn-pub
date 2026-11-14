"""The state an op runs against: the attached debugger, the live run, and debugger tables.

Importing this module loads no model library; the breakpoint table is created on first use.
"""

from __future__ import annotations

from typing import Any


class Context:
    """One debugging session: ``debugger`` (model + tokenizer), ``run`` (may be None), the
    focus address (``frame``/``up``/``down``; None follows the cursor), named checkpoints,
    displays, and the breakpoint/watch table."""

    def __init__(self, debugger: Any = None, run: Any = None):
        self.debugger = debugger
        self.run = run
        self.focus: str | None = None
        self.focus_stack: list[str] = []
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.displays: dict[int, str] = {}
        self._display_id = 1
        self._checkpoint_id = 1
        self._table: Any = None
        # a line an op shows before it finishes (``sweep`` prints its plan); the REPL sets it,
        # None elsewhere (the op's result carries the same information)
        self.notice: Any = None

    @property
    def tokenizer(self) -> Any:
        return getattr(self.debugger, "tokenizer", None)

    @property
    def table(self) -> Any:
        if self._table is None:
            from mdb.engine.breakpoints import Table

            self._table = Table()
        return self._table

    def set_run(self, run: Any) -> None:
        """Replace the live run; checkpoints and watch memory belong to the old one."""
        self.run = run
        self.focus, self.focus_stack = None, []
        self.checkpoints.clear()
        self._checkpoint_id = 1
        if self._table is not None:
            self._table.reset_watches()

    def next_display_id(self) -> int:
        number, self._display_id = self._display_id, self._display_id + 1
        return number

    def next_checkpoint_name(self) -> str:
        while f"c{self._checkpoint_id}" in self.checkpoints:
            self._checkpoint_id += 1
        name = f"c{self._checkpoint_id}"
        self._checkpoint_id += 1
        return name
