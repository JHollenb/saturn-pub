"""Compare the RESULT_JSON of a release-gate job log with an example's validation log.

Usage: compare_result.py NEW_LOG VALIDATION_LOG. Prints the number of numeric/string leaves compared,
the equal count, and every differing leaf (timings, paths, sha256s and revision fields are skipped).
"""
import json
import re
import sys

SKIP = re.compile(r"(second|elapsed|time|wall|_s$|^s$|path|sha256|revision|digest|record_id|^id$|peak|rss|vram"
                  r"|script|seconds|ms$|custody|session|records_file|artifacts|out$|cut$|pointer)", re.I)


def result(path):
    line = [l for l in open(path, errors="replace") if l.startswith("RESULT_JSON ")]
    return json.loads(line[-1][len("RESULT_JSON "):]) if line else None


def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if SKIP.search(str(k)):
                continue
            yield from leaves(v, f"{pre}.{k}" if pre else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, x


new, old = result(sys.argv[1]), result(sys.argv[2])
if new is None or old is None:
    print("COMPARE no RESULT_JSON in", "new" if new is None else "validation")
    sys.exit(0)
a, b = dict(leaves(new)), dict(leaves(old))
common = sorted(set(a) & set(b))
diff = [(k, b[k], a[k]) for k in common if a[k] != b[k]
        and not (isinstance(a[k], float) and isinstance(b[k], float) and abs(a[k] - b[k]) <= 1e-9 * max(1, abs(b[k])))]
print(f"COMPARE leaves compared {len(common)}; equal {len(common) - len(diff)}; differ {len(diff)}; "
      f"only-new {len(set(a) - set(b))}; only-validation {len(set(b) - set(a))}")
for k, o, n in diff[:60]:
    print(f"  DIFF {k}: validation {o!r} -> now {n!r}")
for k in sorted(set(a) - set(b))[:15]:
    print(f"  ONLY-NEW {k}")
for k in sorted(set(b) - set(a))[:15]:
    print(f"  ONLY-VALIDATION {k}")
