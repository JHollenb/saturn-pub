"""Saved native-QK route associations with visually checked primitive scenes.

Every head and case is retained. Ranking is post-outcome exploratory analysis,
not a held-out predictor, a causal localization or an object segmentation model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import runpy
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def analyze(root,destination):
    here=Path(__file__).resolve().parent;data=root/"run-02/extracted/six-areas-result"
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    audit=read(root/"analysis-v1.json");assert audit["mechanics_status"]=="verified"
    observed,tokens=read(data/"observed.json"),read(data/"tokens.json")
    helpers=runpy.run_path(str(here/"analyze_flux_concept_routes.py"));typed=runpy.run_path(str(here/"analyze_flux_mlp_forecast.py"))
    inventory=runpy.run_path(str(here/"flux_scene_inventory.py"))["pixel_inventory"]
    allmaps={};cases={};headrows=[]
    def arr(record):
        p=data/record["path"];assert sha(p)==record["sha256"];return np.load(p,allow_pickle=False)
    for name,record in observed.items():
        qr=record["joint4_sources"];q,k=(arr(qr["qkv"][s]).astype(np.float64)[0] for s in ("q","k"))
        assert q.shape==k.shape==(1536,24,128)
        groups={s:tokens[name]["groups"][s] for s in ("sphere","cube","red","blue","padding")}
        maps={s:np.zeros((24,1024),np.float32) for s in groups}
        keys=k.transpose(1,2,0)
        for first in range(0,1024,64):
            score=q[512+first:512+first+64].transpose(1,0,2)@keys/np.sqrt(128.)
            score-=score.max(-1,keepdims=True);p=np.exp(score);p/=p.sum(-1,keepdims=True)
            for group,ix in groups.items():maps[group][:,first:first+64]=p[...,ix].sum(-1)
        allmaps[name]=maps
        ids=helpers["slot_array"](data/"store",record["joint4_entry"],"img_ids",typed)[0]
        assert ids.shape==(1024,4) and set(map(tuple,ids[:,1:3]))=={(h,w) for h in range(32) for w in range(32)}
        im=np.asarray(Image.open(data/(name+".png"))).copy();inv,labels=inventory(im)
        color_components={}
        for color in ("red","blue"):
            choices=[c for c in inv["components"] if c["color"]==color]
            assert choices,"color instrument cannot identify the controlled component"
            color_components[color]=max(choices,key=lambda c:c["area"])
        pix=labels[(ids[:,1]*16+8).astype(int),(ids[:,2]*16+8).astype(int)]
        color_masks={color:pix==c["component_label"] for color,c in color_components.items()}
        # Manual shape identities were checked on these four natural images.
        # The pixel instrument contributes masks, not these semantic labels.
        shape_color=dict(sphere="red" if name[1]=="0" else "blue",cube="blue" if name[1]=="0" else "red")
        shape_masks={shape:color_masks[color] for shape,color in shape_color.items()}
        scores=[]
        for head in range(24):
            row=dict(head=head)
            for kind,words,masks in (("shape",("sphere","cube"),shape_masks),("color",("red","blue"),color_masks)):
                w0,w1=words;m0,m1=masks[w0],masks[w1]
                row[kind]=dict(first_word_in_own=float(maps[w0][head,m0].mean()),first_word_in_other=float(maps[w0][head,m1].mean()),
                    second_word_in_own=float(maps[w1][head,m1].mean()),second_word_in_other=float(maps[w1][head,m0].mean()))
                row[kind]["selectivity"]=(row[kind]["first_word_in_own"]-row[kind]["first_word_in_other"]+row[kind]["second_word_in_own"]-row[kind]["second_word_in_other"])/2
            scores.append(row)
        cases[name]=dict(scores=scores,component_token_centers={k:int(v.sum()) for k,v in color_masks.items()},
            manual_shape_color_assignment=shape_color,pixel_components=color_components,QK_sha256={s:qr["qkv"][s]["sha256"] for s in ("q","k")})
    for head in range(24):
        row=dict(head=head)
        for kind in ("shape","color"):
            values={name:case["scores"][head][kind]["selectivity"] for name,case in cases.items()}
            row[kind]=dict(values=values,mean=float(np.mean(list(values.values()))),min=min(values.values()),positive_cases=sum(v>0 for v in values.values()))
        headrows.append(row)
    ranks={kind:[r["head"] for r in sorted(headrows,key=lambda r:-r[kind]["mean"])] for kind in ("shape","color")}
    report=dict(schema="mdb-binding-route-association-v1",source_job=audit["job_id"],audit_sha256=sha(root/"analysis-v1.json"),
        cases=cases,heads=headrows,ranks=ranks,new_model_calls=0,
        mechanics_status="valid",instrument_status="calculated softmax from native positioned QK; controlled chromatic masks with manual shape assignment",terminal_status="unassessed",
        scope="Post-outcome all-head association in four dependent same-seed scenes. All1024 image queries and1536keys. Key count and V/output projection matter. No held-out prediction or causal head claim; image-plane center-mask alignment is an approximate observation instrument.")
    with destination.open("x") as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    # Raw route overlays use one shared scale per group across all four scenes.
    # They are annotations of measured routes, not edits or predicted outputs.
    names=list(cases);w,hp=256,282;chosen=[(kind,ranks[kind][0]) for kind in ("shape","color")]
    sheet=Image.new("RGB",(4*w,2*hp),"#f5f5f5");draw=ImageDraw.Draw(sheet)
    for row,(kind,head) in enumerate(chosen):
        word="sphere" if kind=="shape" else "red";scale=max(float(allmaps[n][word][head].max()) for n in names)
        for col,name in enumerate(names):
            ids=helpers["slot_array"](data/"store",observed[name]["joint4_entry"],"img_ids",typed)[0];grid=np.zeros((32,32),np.float32)
            grid[ids[:,1].astype(int),ids[:,2].astype(int)]=allmaps[name][word][head]/max(scale,1e-20)
            heat=np.zeros((32,32,3),np.uint8);heat[...,0]=255;heat[...,1]=(255*(1-grid)).astype(np.uint8);heat[...,2]=heat[...,1]
            mask=Image.fromarray((grid*170).astype(np.uint8)).resize((256,256),Image.Resampling.BILINEAR)
            base=Image.open(data/(name+".png")).resize((256,256));overlay=Image.fromarray(heat).resize((256,256),Image.Resampling.BILINEAR)
            panel=Image.composite(overlay,base,mask);sheet.paste(panel,(col*w,row*hp+26));draw.text((col*w+4,row*hp+7),f"{name} J04.h{head:02d} {word}",fill="black")
    sheet.save(destination.parent/"six-areas-route-overlays.png")
    return dict(ranks={k:v[:4] for k,v in ranks.items()},top={kind:headrows[ranks[kind][0]][kind] for kind in ranks},new_model_calls=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.root.resolve(),a.output.resolve()),sort_keys=True))
