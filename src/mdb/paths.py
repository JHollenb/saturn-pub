"""Explicit data roots; installed package directories are never output roots."""
import os
from pathlib import Path


def data_root():
    return Path(os.environ.get("MDB_HOME", Path.home() / ".local/share/mdb")).expanduser()
