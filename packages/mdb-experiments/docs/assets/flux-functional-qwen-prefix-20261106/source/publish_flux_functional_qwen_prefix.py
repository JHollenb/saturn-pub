"""Publish authenticated compact Qwen-prefix scalar evidence, without models.

No raw array, checkpoint, tokenizer, recurrence or neural runtime is loaded.
The externally supplied manifest comes from authenticated compact collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
AUDITOR_SHA256 = "5004786be76e1f576d2df65e6d1225f8fdd41c4a7e09e5ed8a6c23057db78a97"
WORKER_SHA256 = "2a953ca5ae3bc507e6f79797595a0c69c68325df307a63e4d90bfe88142f62b0"
HELPER_SHA256 = "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a"
RETRIEVER_SHA256 = "0fd61c80696b6bfb76532c9a85ea29b094cde97c0675563c42b05992f5576923"
CASES = ("prompt-00", "prompt-01")
ROW_GROUPS = ("full", "attended_query_rows", "padded_query_rows")
COLORS = ("#245c87", "#b75527", "#418348")
SCOPE = ("Two dependent prospective raw prompts: embedding plus 27 own-state Qwen layers and selected h9/h18/h27 conditioning, "
         "followed by the observed native FLUX context_embedder component. Original Torch primitives are shared. "
         "No downstream image, semantics, prompt-control attribution, independent attention-kernel or full FLUX Session claim.")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(8 << 20), b""):
            digest.update(value)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def member(root, name):
    name = Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "unsafe compact member")
    path = root/name
    require(path.is_file() and not path.is_symlink() and all(not parent.is_symlink() for parent in path.parents if parent != root.parent), "nonregular compact member")
    return path


def authenticate(root, expected_manifest):
    require(sha(root/"files.json") == expected_manifest, "authenticated compact manifest differs")
    manifest = read(root/"files.json")
    require({path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == set(manifest)|{"files.json"}, "complete compact inventory differs")
    for name,digest in manifest.items():
        require(not name.endswith((".npy",".safetensors",".pt",".pth")) and sha(member(root,name)) == digest, "compact scalar/source member differs or contains raw tensors")
    audit,request,plan,summary = [read(member(root,name)) for name in ("audit.json","request.json","plan.json","summary.json")]
    unsigned = {key:value for key,value in request.items() if key!="request_sha256"}
    require(hashlib.sha256(json.dumps(unsigned,sort_keys=True,allow_nan=False).encode()).hexdigest() == request["request_sha256"], "audit request seal differs")
    require(audit["schema"] == "mdb-functional-qwen-prefix-audit-v1" and audit["mechanics_status"] == "authenticated-saved-Qwen-prefix-audit"
            and audit["auditor_sha256"] == AUDITOR_SHA256 and audit["worker_sha256"] == WORKER_SHA256 and audit["helper_sha256"] == HELPER_SHA256
            and audit["new_neural_forward_calls"] == 0, "qualified saved-only audit source/scope differs")
    require(request["source_hashes"]["audit_flux_functional_qwen_prefix.py"] == AUDITOR_SHA256 and request["config"] == plan
            and plan["native_manifest_sha256"] == audit["native_manifest_sha256"] and plan["native_request_sha256"] == audit["native_request_sha256"]
            and plan["source_job"] == summary["source_job"] == audit["job_id"] and summary["request_sha256"] == request["request_sha256"]
            and summary["model_calls"] == 0 and summary["records"] == 2, "audit request/native source joins differ")
    native_source,native_plan,native_summary,native_request = [read(member(root,name)) for name in ("source-source.json","source-plan.json","source-summary.json","source-request.json")]
    require(native_source == audit["source"] and native_source["job_id"] == native_summary["job_id"] == audit["job_id"]
            and native_source["request_sha256"] == native_plan["request_sha256"] == native_summary["request_sha256"] == native_request["request_sha256"] == audit["native_request_sha256"]
            and native_source["worker_sha256"] == WORKER_SHA256 and native_summary["counts"] == audit["counts"], "copied native source identities differ")
    require(sha(root/"source-files.json") == audit["native_manifest_sha256"], "original native manifest differs")
    raw = read(root/"source-files.json")
    for name in ("plan.json","source.json","summary.json","request.json","native-contract.json","forecast-cases.json","native-cases.json","context-consumers.json","context-consumer-operands.json","source-inventory.json","timeline.jsonl","runtime-contract.json"):
        require(sha(member(root,"source-"+name)) == raw["files"][name]["sha256"], "copied original metadata hash differs: "+name)
    require(sha(root/"source-plan.json") == native_source["effective_plan_sha256"] and sha(root/"source-request.json") == native_source["request_record_sha256"], "executed plan/request bytes differ")
    for key,row in native_source["retained_metadata"].items():
        path = member(root,Path(native_plan[key]).name)
        require(sha(path) == native_plan[key+"_sha256"] == row["sha256"] == raw["files"][row["path"]]["sha256"], "copied source/recipe/retriever differs")
    require(sha(member(root,"retrieve_evidence.py")) == RETRIEVER_SHA256, "raw checksum retriever differs")
    external = read(member(root,"external-evidence.json"))
    require(external["manifests"]["native"] == {name:row["sha256"] for name,row in raw["files"].items()}, "original remote retrieval identities differ")
    provenance = read(member(root,"catalog/provenance.json"))
    records = list((root/"catalog/equations").glob("*.json"))
    require(provenance["records"] == provenance["dependent_prompts"] == 2 and provenance["family_induction"] is False and len(records) == 2
            and {row["case"] for row in provenance["queries"]} == set(CASES)
            and all(row["exact"] == 1 and row["changed_input"] == row["missing_mask"] == 0 for row in provenance["queries"]), "two exact-context dependent catalog occurrences differ")
    require({read(path)["fingerprint"] for path in records} == {row["fingerprint"] for row in provenance["queries"]}
            and all(path.stem == read(path)["fingerprint"] and read(path)["evidence"]["audit_sha256"] == sha(root/"audit.json") for path in records), "catalog/audit identity differs")
    require(set(audit["accounts"]["cases"]) == set(CASES) and audit["accounts"]["prompt_response_order"] == dict(positive=CASES[1],negative=CASES[0]), "prompt/sign occurrence scope differs")
    return audit,manifest,summary


def field_order(audit):
    fields = audit["accounts"]["cases"][CASES[0]]["fields"]
    return sorted((name for name in fields if name.startswith("hidden.")), key=lambda name:int(name.split(".")[1]))


def group_metric(row, group):
    return row if group == "full" else row[group]["comparison"]


def plot_line(ax, values, label, color, style="-"):
    ax.plot(range(len(values)), [float("nan") if value is None else value for value in values], label=label,color=color,linestyle=style,marker="o",markersize=2.8,linewidth=1.3)


def unsigned_axis(ax):
    finite = [float(value) for line in ax.lines for value in line.get_ydata() if value == value]
    ax.set_ylim(bottom=0)
    if not finite or not any(finite):
        ax.set_ylim(0,1)
        ax.set_yticks([0])
        ax.text(.5,.5,"No defined values" if not finite else "All plotted values are exactly zero",ha="center",va="center",transform=ax.transAxes,fontsize=9)


def depth_axis(ax, fields, ylabel):
    n = len(fields)
    ticks = sorted({0,n-1,*range(0,n,max(1,n//7))})
    ax.set_xticks(ticks,["Embedding" if i == 0 else "h"+str(i) for i in ticks],rotation=30,ha="right")
    ax.set_xlim(-.4,n-.6)
    ax.set_xlabel("Complete hidden-state index (native interior convention)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)


def finish(fig,path):
    fig.set_dpi(160)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    rows = []
    for index,ax in enumerate(fig.axes):
        b = ax.get_tightbbox(renderer)
        rows.append(dict(axis=index,within_figure=bool(b.x0>=box.x0-1 and b.y0>=box.y0-1 and b.x1<=box.x1+1 and b.y1<=box.y1+1),bbox=[float(value) for value in b.bounds]))
    require(all(row["within_figure"] for row in rows), "scientific figure labels exceed canvas")
    fig.savefig(path,dpi=160)
    plt.close(fig)
    return rows


def figures(audit,output):
    fields = field_order(audit)
    qa = {}
    fig,axes = plt.subplots(2,2,figsize=(14,9),layout="constrained")
    for case_index,case in enumerate(CASES):
        for column,key in enumerate(("rms","relative_rms")):
            ax = axes[case_index,column]
            for group,color,style,label in zip(ROW_GROUPS,COLORS,("-","--",":"),("All query rows","Attended query rows","Padded query rows"),strict=True):
                values = []
                for field in fields:
                    row = audit["accounts"]["cases"][case]["fields"][field]
                    metric = row["full"] if group == "full" else group_metric(row,group)
                    values.append(None if metric is None else metric[key])
                plot_line(ax,values,label,color,style)
            ax.set_title("Prompt "+str(case_index)+" · "+("absolute error" if column == 0 else "native-relative error"))
            depth_axis(ax,fields,"RMS(functional − native), activation units" if column == 0 else "Error RMS / same native field RMS")
            unsigned_axis(ax)
            ax.legend(fontsize=8)
    fig.suptitle("Qwen prefix: complete-state prediction errors\nAll query rows remain primary; attended/padded rows are diagnostics. Linear axes preserve zeros.",fontsize=13)
    qa["hidden-errors.png"] = finish(fig,output/"hidden-errors.png")
    fig,axes = plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for column,field in enumerate(("conditioning","projected")):
        for row_index,key in enumerate(("rms","relative_rms")):
            ax = axes[row_index,column]
            for case_index,case in enumerate(CASES):
                row = audit["accounts"]["cases"][case]["fields"][field]
                values = [row["full"][key],*[None if row[group]["comparison"] is None else row[group]["comparison"][key] for group in ROW_GROUPS[1:]]]
                plot_line(ax,values,"Prompt "+str(case_index),COLORS[case_index])
            ax.set_xticks(range(3),["All rows","Attended","Padded"])
            ax.set_title(("E = cat(interior states)" if field == "conditioning" else "Context projection · native consumer reference"))
            ax.set_ylabel("RMS(functional − native), field units" if key == "rms" else "Error RMS / same native field RMS")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            unsigned_axis(ax)
    fig.suptitle("Conditioning and first component port\nFields have separate native denominators; no downstream image prediction.",fontsize=13)
    qa["conditioning-errors.png"] = finish(fig,output/"conditioning-errors.png")
    response = audit["accounts"]["signed_prompt_response"]
    fig,axes = plt.subplots(3,2,figsize=(14,12),layout="constrained")
    for key,label,color,style in (("native_rms","Native",COLORS[0],"-"),("predicted_rms","Functional",COLORS[1],"--")):
        plot_line(axes[0,0],[response[field][key] for field in fields],label,color,style)
    for key,label,color,style in (("native_signed_mean","Native",COLORS[0],"-"),("predicted_signed_mean","Functional",COLORS[1],"--")):
        plot_line(axes[1,0],[response[field][key] for field in fields],label,color,style)
    for ax,key,title,ylabel in ((axes[0,1],"rms","Response prediction error","RMS(functional response − native response)"),
        (axes[1,1],"relative_rms","Native-response relative error","Error RMS / native prompt-response RMS"),
        (axes[2,0],"cosine","Full response vector agreement","Cosine (undefined when a response has zero RMS)")):
        plot_line(ax,[response[field][key] for field in fields],"Complete response",COLORS[2])
        ax.set_title(title)
        depth_axis(ax,fields,ylabel)
        if key == "cosine":
            ax.set_ylim(-1.05,1.05)
            ax.axhline(0,color="#777777",linewidth=.6)
        else:
            unsigned_axis(ax)
    axes[0,0].set_title("Prompt-response magnitudes (RMS is unsigned)")
    axes[1,0].set_title("Signed prompt-response means")
    depth_axis(axes[0,0],fields,"RMS(prompt 1 − prompt 0), activation units")
    depth_axis(axes[1,0],fields,"Mean(prompt 1 − prompt 0), signed activation units")
    unsigned_axis(axes[0,0])
    axes[1,0].axhline(0,color="#777777",linewidth=.6)
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=8)
    axes[2,1].axis("off")
    table = [[label,*["undefined" if response[field][key] is None else f"{response[field][key]:.6g}" for key in ("native_rms","rms","relative_rms","cosine")]] for field,label in (("conditioning","E"),("projected","Context"))]
    tab = axes[2,1].table(cellText=table,colLabels=["Field","Native RMS","Error RMS","Relative","Cosine"],loc="center",cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1,1.8)
    axes[2,1].set_title("Conditioning / component response endpoints",pad=15)
    fig.suptitle("Signed prompt 1 − prompt 0 response, without token alignment\nSame coordinate positions are compared; they are not lexical correspondences or a controlled semantic contrast.",fontsize=13)
    qa["prompt-responses.png"] = finish(fig,output/"prompt-responses.png")
    return qa


def publish(root,output,*,expected_manifest,qualification,collection_receipt=None):
    root,output,qualification = Path(root).resolve(),Path(output).resolve(),Path(qualification).resolve()
    audit,manifest,summary = authenticate(root,expected_manifest)
    qual = read(qualification)
    require(qual["status"] == "passed" and qual["source_sha256"]["qualification/audit_flux_functional_qwen_prefix.py"] == AUDITOR_SHA256, "qualified audit receipt source differs")
    require(not output.exists(), "publication output exists")
    output.mkdir(parents=True)
    shutil.copytree(root,output/"compact-evidence")
    shutil.copyfile(qualification,output/"audit-qualification.json")
    shutil.copyfile(root/"audit.json",output/"audit-data.json")
    (output/"source").mkdir()
    for name,digest in (("audit_flux_functional_qwen_prefix.py",AUDITOR_SHA256),("flux_functional_qwen_prefix_worker.py",WORKER_SHA256),("flux_functional_qwen_prefix.py",HELPER_SHA256)):
        require(sha(HERE/name) == digest,"source snapshot differs")
        shutil.copyfile(HERE/name,output/"source"/name)
    shutil.copyfile(__file__,output/"source"/Path(__file__).name)
    portable = dict(schema="mdb-functional-qwen-prefix-publication-v1",scope=SCOPE,
        orientation="Functional minus native; prompt responses are prompt 1 minus prompt 0 within each lane.",
        denominator="Corresponding native field RMS, or native prompt1-minus-prompt0 response RMS. Zero denominator is null; RMS is unsigned, signed means are separate.",
        source=dict(native_job=audit["job_id"],native_request_sha256=audit["native_request_sha256"],native_manifest_sha256=audit["native_manifest_sha256"],audit_job=summary["job_id"],
            audit_sha256=sha(root/"audit.json"),audit_manifest_sha256=expected_manifest,auditor_sha256=AUDITOR_SHA256,worker_sha256=WORKER_SHA256,helper_sha256=HELPER_SHA256,qualification_sha256=sha(qualification)),
        accounts=audit["accounts"],parser=audit["parser"],coverage=audit["coverage"],census_scope=audit["census_scope"],counts=audit["counts"],
        field_counts=dict(primary_full_fields=sum(len(row["fields"]) for row in audit["accounts"]["cases"].values()),
            primary_scalar_positions=sum(value["full"]["sample_count"] for row in audit["accounts"]["cases"].values() for value in row["fields"].values()),
            derived_prompt_response_fields=len(audit["accounts"]["signed_prompt_response"]),
            derived_prompt_response_positions=sum(row["sample_count"] for row in audit["accounts"]["signed_prompt_response"].values()),
            overlap=True,scope="Positions repeat across layers and selected conditioning; derived diagnostics overlap primary fields. These are not independent observations."),
        native_contract=audit["native_contract"],runtime_contract=audit["runtime_contract"],parameter_roles=audit["parameter_roles"],parameter_restoration=audit["parameter_restoration"],
        catalog=read(root/"catalog/provenance.json"),retrieval=read(root/"external-evidence.json"),
        limits=["Two dependent prompt occurrences, not an independent family or semantic-control experiment.",
            "The saved CPU audit reapplied the same pinned tokenizer/parser; it performed zero neural calls, not zero tokenizer calls. This publication does not parse prompts.",
            "All 512 query rows are retained and consumed; padding rows are not discarded.","Selected attention kernel and original native strides are not independently observed by this audit.",
            "The declared source-bound root-call census is distinct from actual two native output records and four observed component operands; full Qwen submodule census is unrecorded.",
            "Full hidden/E/component arrays were audited near data; this publication contains scalars, references and checksums, not raw states or weights.",
            "No downstream image, VAE, denoiser, complete FLUX Session or semantic attribution is established."])
    save(output/"publication.json",portable)
    qa = figures(audit,output)
    save(output/"figure-layout.json",qa)
    (output/"README.md").write_text("# Qwen prefix saved evidence\n\n"+SCOPE+"\n\nErrors are functional minus native. Prompt responses are prompt 1 minus prompt 0, compared at original token-coordinate positions without lexical alignment. RMS is unsigned; signed means and every first discrepancy/tail remain in publication.json. Each relative value uses its same native field or native response denominator. Zero-denominator values remain undefined, not zero. Linear axes preserve exact zeros. Different layers/fields have different native amplitudes; these plots do not identify causal amplification.\n\nAll full array references, two dependent catalog occurrences, exact source/qualification, parameter recipes and checksum retrievers are in compact-evidence/. No raw state, pretrained bank, image or model execution is included.\n")
    outputs = {path.relative_to(output).as_posix():sha(path) for path in output.rglob("*") if path.is_file()}
    receipt = dict(schema="mdb-functional-qwen-prefix-publication-receipt-v1",publisher_sha256=sha(__file__),compact_root=str(root),authenticated_manifest_sha256=expected_manifest,
        authentication_scope="Expected compact manifest supplied by authenticated external collection; every compact member and internal source/request/catalog/recipe join checked.",
        inputs={"files.json":expected_manifest,**manifest},qualification=dict(path=str(qualification),sha256=sha(qualification)),
        collection_receipt=None if collection_receipt is None else dict(path=str(collection_receipt),sha256=sha(collection_receipt)),outputs=outputs,
        raw_arrays_loaded=False,weights_loaded=False,new_model_calls=0,new_numerical_recurrence=0,catalog_records=2)
    save(output/"publication-receipt.json",receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-manifest",required=True)
    parser.add_argument("--qualification",type=Path,required=True)
    parser.add_argument("--collection-receipt",type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.pack,args.out,expected_manifest=args.expected_manifest,qualification=args.qualification,collection_receipt=args.collection_receipt),sort_keys=True))
