"""Typed reads of stored (untrusted) data: the one place a view asks "is this really a ...?".

A stored body is whatever JSON a writer put there; a view never assumes a field's type. A string
field is used only when it is a ``str``, a mapping only when it is a ``dict``, a list only when it
is a ``list``, a number only when it is a finite, non-bool ``int``/``float``.
"""
from __future__ import annotations


def as_str(value, default: str = "") -> str:
    return value if isinstance(value, str) else default


def as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def is_number(value) -> bool:
    """A finite int/float that is not a bool."""
    # the range test is False for NaN/inf and keeps float() of an int from overflowing
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and -1e300 < value < 1e300


def as_number(value, default=None):
    return value if is_number(value) else default


def as_int(value, default=None):
    return value if isinstance(value, int) and not isinstance(value, bool) else default
