"""x86-64 binaries as an mdb machine, executed by torchx86 in batched lanes (see ``binary``).

``attach(program_dir, input)`` opens a debugger session (``session.BinRun``) with every verb."""
from .binary import Binary, torchx86


def attach(*args, **kwargs):
    from .session import attach as _attach  # noqa: PLC0415  (torch only when used)

    return _attach(*args, **kwargs)


__all__ = ["Binary", "attach", "torchx86"]
