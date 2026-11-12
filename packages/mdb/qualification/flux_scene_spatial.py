"""Output-assisted geometric windows over actual FLUX image IDs, not latent semantics."""
from __future__ import annotations


def make_rows():
    rows = [{"id":"base","kind":"reference","history":"s0c0r0"},
            {"id":"sham","kind":"sham","history":"s0c0r0"}]
    for factor,history in (("shapeA","s1c0r0"),("colorB","s0c1r0"),("relation","s0c0r1")):
        rows += [{"id":factor+"."+mode,"kind":"spatial","history":"s0c0r0",
                  "source":history,"factor":factor,"mode":mode}
                 for mode in ("full","left","right","both","outside","shifted")]
    return rows


def supports(base,source,img_ids,*,height,width,pad=16):
    """Union historical base/source component bounds by spatial site, one-cell halo.

    H/W IDs determine geometry even when physical row order is permuted. Pixel
    footprints locate windows; model receptive fields can extend beyond them.
    """
    import numpy as np

    ids = np.asarray(img_ids)
    ids = ids[0] if ids.ndim==3 and ids.shape[0]==1 else ids
    if ids.ndim!=2 or ids.shape[1]!=4 or not np.isfinite(ids).all() or not np.equal(ids,np.round(ids)).all():
        raise ValueError("finite native integer T/H/W/L IDs required")
    coords = [tuple(int(x) for x in row) for row in ids]
    hs,ws = sorted({c[1] for c in coords}),sorted({c[2] for c in coords})
    if (len({c[0] for c in coords})!=1 or len({c[3] for c in coords})!=1
            or hs!=list(range(len(hs))) or ws!=list(range(len(ws)))
            or len(set(coords))!=len(coords) or len(coords)!=len(hs)*len(ws)
            or len(hs)%2 or height<=0 or width<=0 or pad<0):
        raise ValueError("complete even-height rectangular native image grid required")
    panels = []
    for inventory in (base,source):
        components = sorted(inventory["components"],key=lambda c:c["centroid"][0])
        if len(components)!=2 or not components[0]["centroid"][0]<width/2<components[1]["centroid"][0]:
            raise ValueError("two historical components at distinct left/right sites required")
        panels.append(components)
    windows,masks = {},{}
    for site,index in (("left",0),("right",1)):
        bounds = [p[index]["bounds"] for p in panels]
        box = [max(0,min(b[0] for b in bounds)-pad),max(0,min(b[1] for b in bounds)-pad),
               min(width,max(b[2] for b in bounds)+pad),min(height,max(b[3] for b in bounds)+pad)]
        windows[site] = box
        selected = []
        for i,c in enumerate(coords):
            x,y = (c[2]+.5)*width/len(ws),(c[1]+.5)*height/len(hs)
            if box[0]<=x<box[2] and box[1]<=y<box[3] and (x<width/2 if site=="left" else x>=width/2):
                selected.append(i)
        if not selected:
            raise ValueError("empty historical window")
        masks[site] = selected
    if set(masks["left"]) & set(masks["right"]):
        raise ValueError("independent spatial windows overlap")
    both = set(masks["left"]) | set(masks["right"])
    lookup = {(c[1],c[2]):i for i,c in enumerate(coords)}
    masks.update(both=sorted(both),outside=sorted(set(range(len(coords)))-both),
        shifted=sorted(lookup[((coords[i][1]+len(hs)//2)%len(hs),coords[i][2])] for i in both),
        full=list(range(len(coords))),sham=[])
    return {"masks":masks,"windows":windows,"pad_pixels":pad,"native_grid":[len(hs),len(ws)],
        "selected_rows":{k:len(v) for k,v in masks.items()},
        "shifted_union_overlap":len(set(masks["shifted"]) & both),
        "policy":"Historical base/source RGB component bounding-box unions by left/right site,16px halo clipped at midline; native T/H/W/L IDs map cell centers; shifted control is half-height cyclic shift; no new patch outcome used",
        "semantic_scope":"Supplied output-assisted geometric windows; no latent-derived object identities or pure semantic receptive fields"}


def patch_latent(current,source,indices):
    """Preserve native dtype and every unwritten row exactly."""
    if current.shape!=source.shape or current.ndim!=3 or current.shape[0]!=1 or current.dtype!=source.dtype:
        raise ValueError("matched native B1 latent tensors required")
    if len(set(indices))!=len(indices) or any(type(i) is not int or i<0 or i>=current.shape[1] for i in indices):
        raise ValueError("unique in-range native row indices required")
    value = current.clone()
    if indices:
        value[:,indices,:] = source[:,indices,:]
    return value
