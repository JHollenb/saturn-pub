"""Offline native Qwen: causal layer intervention, two futures, commit and restore."""

import json
from pathlib import Path

from saturn_pub import Act
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.store import LocalStore

adapter = QwenAdapter.tiny()
session = adapter.session([5, 7, 11])
session.continue_(2)
parent = session.capture()
native, candidate = session.fork(parent), session.fork(parent)
candidate.apply(Act.zero("hidden"))
adapter.generate(native, tokens=2)
adapter.generate(candidate, tokens=2)
report = {
    "native": native.read("tokens").tolist(),
    "candidate": candidate.read("tokens").tolist(),
    "same_parent": candidate._origin == native._origin,
}
session.commit(candidate)
report["restore"] = session.restore(parent).to_dict()
store = LocalStore("outputs/ar")
report["saved_parent"] = store.save(parent)
for receipt in candidate.receipts:
    store.receipt(receipt)
Path("outputs/ar/report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
