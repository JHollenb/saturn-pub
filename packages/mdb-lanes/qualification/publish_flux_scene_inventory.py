"""Publish generic scene evidence records and plots from audited observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")


def publish(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw

    run = root/json.loads((root/"selected-run.json").read_text())["run"]
    analysis = json.loads((run/"analysis-v1.json").read_text())
    result = run/"extracted/flux-scene-inventory-result"
    summary, source, plan = (json.loads((result/name).read_text()) for name in ("summary.json","source.json","plan.json"))
    supports = json.loads((result/"supports.json").read_text())
    binding = {"model_identity": source["model_identity"], "checkpoint_sha256": source["checkpoint_sha256"],
               "execution": source["execution"], "seed": plan["seed"], "steps": 4,
               "job_id": summary["job_id"], "analysis_sha256": sha(run/"analysis-v1.json"),
               "source_sha256": sha(result/"source.json"), "seal": summary["seal"],
               "terminal_status": "unassessed", "statistical_unit": "one seed/checkpoint; eight coupled factorial histories and dependent interventions"}
    objects, relations, causal = [], [], []
    for row_name, inventory in analysis["inventories"].items():
        rgb_sha = sha(result/(row_name+".png"))
        for component in inventory["components"]:
            objects.append({"id": rgb_sha[:16]+"."+component["id"], "output_sha256": rgb_sha,
                "row": row_name, "observation": component,
                "text_address": None, "internal_object_address": None,
                "evidence_status": "pixel-observation", "binding": binding,
                "limitations": "Generic chromatic connected component; no automatic latent semantic identity or cross-image identity claim"})
        for relation in inventory["relations"]:
            relations.append({"output_sha256": rgb_sha, "row": row_name, "observation": relation,
                "internal_relation_address": None, "evidence_status": "pixel-centroid-relation", "binding": binding})
    for number,(factor,history) in enumerate((("shapeA","s1c0r0"),("colorB","s0c1r0"),("relation","s0c0r1"))):
        for mode in ("full","top","shifted"):
            name = factor+"."+mode
            row = json.loads((result/"rows"/(name+".json")).read_text())
            causal.append({"id": f"flux.scene.factor_{number}.image-state.{mode}", "binding": binding,
                "supplied_factor_alias": factor, "semantic_operator": None,
                "source": {"cut": row["source_cut"], "prompt": next(h["prompt"] for h in plan["histories"] if h["id"]==history)},
                "recipient": {"cut": row["parent"], "prompt": plan["histories"][0]["prompt"]},
                "address": {"slot": "image", "rows": row["selected_rows"], "clock": "step1 joint4 exit"},
                "update": "coherent observed source image-row replacement; recipient other state retained",
                "consumer": "native remaining denoiser, scheduler, later steps, VAE and final RGB",
                "effects": analysis["effects"][name], "pixel_inventory": analysis["inventories"][name],
                "continuation": row["schedulers"], "evidence_status": "causal-intervention-observation",
                "applicability": "only the exact source/recipient histories and numerical profile; no expanded applicability",
                "counterexamples": {k:v for k,v in analysis["prediction_observations"].items() if k.startswith(factor) and v["observed"] is False},
                "unmeasured_predictions": {k:v for k,v in analysis["prediction_observations"].items() if k.startswith(factor) and v["observed"] is None}})
    save(root/"scene-records.json", {"schema": "mdb-scene-inventory-records-v1", "binding": binding,
        "objects": objects, "relations": relations, "causal_records": causal,
        "automatic_semantic_decomposition": False, "automatic_computation_atlas_ingestion": False,
        "unknown_scene_fields": ["latent object identity","shape semantics","occluded geometry","lighting source","material","shadow source","reflections","background provenance","general cross-context applicability"]})
    figures = root/"figures"
    figures.mkdir(exist_ok=False)
    boundaries = ("early","entry","attention","mlp","post")
    terms = ("shapeA","colorB","relation","shapeA*colorB","shapeA*relation","colorB*relation","shapeA*colorB*relation")
    matrix = np.array([[analysis["state_relationships"][b]["image"]["coefficients"][t]["rms"]
                        if b in ("early","entry","post") else analysis["state_relationships"][b]["coefficients"][t]["rms"]
                        for t in terms] for b in boundaries])
    fig,ax = plt.subplots(figsize=(12,4.6))
    im = ax.imshow(np.log10(np.maximum(matrix,1e-12)),aspect="auto",cmap="viridis")
    ax.set_xticks(range(len(terms)),[t.replace("*"," × ") for t in terms],rotation=24,ha="right")
    ax.set_yticks(range(5),["step0 single4 image","step1 joint4 entry image","joint4 attention return","joint4 MLP return","joint4 exit image"])
    for i in range(5):
        for j in range(7):
            ax.text(j,i,f"{matrix[i,j]:.3g}",ha="center",va="center",color="white",fontsize=9)
    fig.colorbar(im,ax=ax,label="log10 coefficient RMS; each row has its own tensor units")
    ax.set_title("Observed scene-state factorial effects and interactions")
    fig.tight_layout()
    fig.savefig(figures/"state-relationships.png",dpi=150)
    plt.close(fig)
    base = Image.open(result/"native.s0c0r0.png").convert("RGB")
    panel = Image.new("RGB",(3*512,550),"white")
    draw = ImageDraw.Draw(panel)
    for i,factor in enumerate(("shapeA","colorB","relation")):
        overlay = Image.new("RGBA",base.size,(0,0,0,0))
        painter = ImageDraw.Draw(overlay)
        coordinates = supports[factor]["coordinates"]
        heights,widths = sorted({p[0] for p in coordinates}),sorted({p[1] for p in coordinates})
        for index in supports[factor]["top"]:
            h,w = coordinates[index]
            y,x = heights.index(h),widths.index(w)
            painter.rectangle([round(x*512/len(widths)),round(y*512/len(heights)),
                round((x+1)*512/len(widths)),round((y+1)*512/len(heights))],fill=(255,210,0,80))
        combined = Image.alpha_composite(base.convert("RGBA"),overlay).convert("RGB")
        painter = ImageDraw.Draw(combined)
        for component in analysis["inventories"]["native.s0c0r0"]["components"]:
            painter.rectangle(component["bounds"],outline="black",width=2)
        panel.paste(combined,(i*512,38))
        draw.text((i*512+8,10),factor+" prefix-selected support; boxes measured later",fill="black")
    panel.save(figures/"supports-and-objects.png")
    save(root/"publication.json", {"binding": binding, "scene_records_sha256": sha(root/"scene-records.json"),
        "pixel_object_records": len(objects), "pixel_relation_records": len(relations), "causal_records": len(causal),
        "figures": {p.name:sha(p) for p in sorted(figures.iterdir())}})
    return {"objects":len(objects),"relations":len(relations),"causal_records":len(causal),"figures":str(figures)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    print(json.dumps(publish(p.parse_args().root),sort_keys=True))
