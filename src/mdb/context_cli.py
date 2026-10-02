"""Dependency-light command parsing for the optional typed Thott reader."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb context", description="Inspect and verify typed context registers")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "verify"):
        child = sub.add_parser(command)
        child.add_argument("register", type=Path)
    child = sub.add_parser("diff")
    child.add_argument("register", type=Path)
    child.add_argument("left_node")
    child.add_argument("right_node")
    parser.parse_args(argv)
    try:
        from .context_register import main as implementation
    except ModuleNotFoundError as error:
        if error.name in {"torch", "safetensors", "numpy"}:
            parser.error("typed context register operations require mdb-runtime[autoregressive]")
        raise
    return implementation(argv)
