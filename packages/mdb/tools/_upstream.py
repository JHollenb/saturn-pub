"""Explicit source selection for maintainer-only extraction tools."""
import argparse
from pathlib import Path


def upstream_checkout():
    parser = argparse.ArgumentParser(description="Adapt an explicitly selected Saturn source.")
    parser.add_argument("--upstream", type=Path, required=True,
                        help="Path to the Saturn research checkout (not saturn-pub)")
    source = parser.parse_args().upstream.expanduser().resolve()
    if not (source / "src/saturn").is_dir():
        parser.error("upstream must contain src/saturn")
    return source
