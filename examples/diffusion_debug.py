"""Offline random UNet: branch a DDIM trajectory, intervene, decode native output."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from saturn_pub import Act
from saturn_pub.adapters.diffusion import DiffusionAdapter
from saturn_pub.store import LocalStore

adapter = DiffusionAdapter.tiny()
session = adapter.session(steps=4)
session.continue_()
parent = session.capture()
native, candidate = session.fork(parent), session.fork(parent)
candidate.apply(Act.zero("latent"))
native.continue_(3)
candidate.continue_(3)
output = Path("outputs/diffusion")
output.mkdir(parents=True, exist_ok=True)
for name, branch in (("native", native), ("candidate", candidate)):
    image = adapter.decode(branch)[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(output / f"{name}.png")
store = LocalStore(output)
report = {
    "parent": store.save(parent),
    "random_model": True,
    "latent_rms_difference": float(
        (native.read("latent") - candidate.read("latent")).square().mean().sqrt()
    ),
    "restore": session.restore(parent).to_dict(),
}
(output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
