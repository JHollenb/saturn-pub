"""``help`` lists the slots that are not built yet with their text, plainly."""

import io

from mdb.ops import Context, registry
from mdb.repl.shell import Shell


def test_help_prints_each_unbuilt_slots_text_plainly():
    shell = Shell(Context(None), out=io.StringIO())
    shell.execute("help")
    line = next(l for l in shell.out.getvalue().splitlines() if l.startswith("not built yet:"))
    slots = {n: o.milestone for n, o in registry().items() if o.milestone is not None}
    assert slots and all(f"{name}: {text}" in line for name, text in slots.items())
    assert not any(f"{name}(" in line for name in slots)
