"""Authenticate and prepare saved binding support/complement source states.

No model construction, execution, or prior-artifact mutation. Frameworks are
loaded only by the explicit source-preparation call. Supplied historical states
and images are observations, not prospective numerical forecasts.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "mdb-flux-binding-support-sources-v1"
HELPERS = {}
ARMS = ("whole_Q", "support_P", "complement_P", "whole_P")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request_identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False, default=lambda item: dict(item) if isinstance(item, Mapping) else _unsupported(item))
        stream.write("\n")


def _unsupported(item):
    raise TypeError("unsupported metadata: " + type(item).__name__)


class Evidence:
    """Verify every consumed member, including all public StateCut dependencies."""
    def __init__(self, root, manifest_sha256, expected_request):
        self.root = Path(root).resolve(); self.consumed = {}
        require(sha(self.root / "files.json") == manifest_sha256, "source inventory differs")
        self.manifest_sha256 = manifest_sha256
        self.inventory = json.loads((self.root / "files.json").read_text())
        require(isinstance(self.inventory, dict) and self.inventory, "nonempty inventoried source required")
        request = self.read("request.json")
        require(request.get("request_sha256") == request_identity(request) == expected_request, "source request identity differs")
        self.request = request
        require(self.read("plan.json") == request["config"], "source plan/request differs")

    def member(self, name):
        require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe source member")
        path = self.root / name
        require(path.resolve().is_relative_to(self.root) and not any(item.is_symlink() for item in (path, *path.parents)), "escaped/symlinked source member")
        require(name in self.inventory and path.is_file(), "uninventoried source member: " + name)
        expected = self.inventory[name]
        wanted = expected["sha256"] if isinstance(expected, dict) else expected
        actual = sha(path)
        require(actual == wanted and (not isinstance(expected, dict) or "bytes" not in expected or path.stat().st_size == expected["bytes"]), "source member differs: " + name)
        self.consumed[name] = dict(path=name, sha256=actual, bytes=path.stat().st_size)
        return path

    def read(self, name):
        return json.loads(self.member(name).read_text())

    def cut(self, fingerprint):
        from saturn_pub.store import LocalStore

        require(isinstance(fingerprint, str) and len(fingerprint) == 64 and all(value in "0123456789abcdef" for value in fingerprint), "invalid StateCut fingerprint")
        metadata = self.read("store/cuts/"+fingerprint+".json")
        for page in metadata["slots"].values():
            self.member("store/pages/"+page+".json")
        for blob in metadata["physical_blobs"]:
            self.member("store/blobs/"+blob+".safetensors")
        return LocalStore(self.root / "store").load(fingerprint, device="cpu")

    def array(self, record):
        import numpy as np
        import torch

        path = self.member(record["path"])
        require(sha(path) == record["sha256"] and record.get("dtype") == "bfloat16" and record.get("storage_dtype") == "uint16", "source BF16 array descriptor differs")
        array = np.load(path, allow_pickle=False)
        require(array.dtype == np.uint16 and list(array.shape) == record["shape"], "source BF16 array layout differs")
        value = torch.from_numpy(array.copy()).view(torch.bfloat16)
        require(bool(torch.isfinite(value).all()), "nonfinite source array")
        return value

    def receipt(self):
        return dict(root=str(self.root), files_manifest_sha256=self.manifest_sha256, request_sha256=self.request["request_sha256"],
            verification_scope="Consumed inventoried members and complete consumed StateCut closure", consumed_members=dict(sorted(self.consumed.items())))


def save_tensor(output, relative, value):
    """Same rich descriptor ABI as the frozen raw-to-image tensor serializer."""
    import numpy as np
    import torch

    require(value.device.type == "cpu", "CPU witness required")
    materialized = value.detach().contiguous()
    stored = materialized.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else materialized.numpy()
    path = Path(output) / relative; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, stored, allow_pickle=False)
    return dict(path=relative, sha256=sha(path), bytes=path.stat().st_size, content_sha256=hashlib.sha256(stored.tobytes(order="C")).hexdigest(),
        shape=list(value.shape), stride=list(value.stride()), logical_dtype=str(value.dtype), stored_dtype=str(stored.dtype),
        encoding="bf16-bits-v1" if value.dtype == torch.bfloat16 else "native-numpy-v1", device_at_capture="cpu")


def slots(payload):
    from saturn_pub.values import describe

    return {name: describe(value) for name, value in payload.items()}


def cut_record(cut):
    return dict(cut=cut.fingerprint, parent=cut.parent, model_identity=cut.model_identity, execution=dict(cut.execution),
        point=dict(cut.execution_point), boundary=cut.boundary, payload_descriptors=slots(cut.payload))


def require_pair(p, q):
    require(p.model_identity == q.model_identity and p.execution == q.execution and p.execution_point == q.execution_point, "paired model/profile/full clock differs")
    left, right = slots(p.payload), slots(q.payload)
    require(set(left) == set(right) and {"image", "text", "conditioning"} <= set(left), "paired state slots differ")
    require(all(left[name] == right[name] for name in left if name not in ("image", "text")), "paired non-I/T slot differs")
    return sorted(name for name in left if name not in ("image", "text"))


def require_clock(cut, adapter, index, *, step=2, phase="joint"):
    require(cut.model_identity == adapter.model_identity, "source model identity differs from current adapter")
    point = dict(cut.execution_point)
    require(point == adapter.point(cut.payload).to_dict(), "source full execution point differs from current adapter")
    require(point["logical_step"] == step and point["phase"] == phase and point["index"] == index, "source boundary clock differs")


def same_bits(left, right):
    import torch

    return left.shape == right.shape and left.dtype == right.dtype == torch.bfloat16 and torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16))


def compare(left, right, rows=None):
    import torch

    if rows is not None:
        left, right = left[:, rows], right[:, rows]
    delta = left.double()-right.double()
    require(delta.numel() > 0 and bool(torch.isfinite(delta).all()), "finite nonempty comparison required")
    bit_dtype = {torch.bfloat16: torch.int16, torch.float64: torch.int64}.get(left.dtype)
    bits = left.contiguous().view(bit_dtype) != right.contiguous().view(bit_dtype) if left.dtype == right.dtype and bit_dtype is not None else delta != 0
    changed = torch.any(bits, dim=-1)[0]
    selected = list(range(left.shape[1])) if rows is None else rows
    return dict(elements=delta.numel(), rms=float(delta.square().mean().sqrt()), l2=float(delta.norm()), max_abs=float(delta.abs().max()),
        signed_mean=float(delta.mean()), numeric_nonzero=int(torch.count_nonzero(delta)), raw_bit_changed=int(torch.count_nonzero(bits)),
        signed_zero_defects=int(torch.count_nonzero((left == 0) & (right == 0) & (torch.signbit(left) != torch.signbit(right)))),
        changed_rows=[row for row, different in zip(selected, changed.tolist(), strict=True) if different])


def prepare_binding_sources(config, output, store, adapter):
    """Return CPU source cuts and exact BF16 text row replacements per case.

    Config keys: binding_{source,route}_{root,manifest_sha256,request_sha256},
    binding_plan (literal next-cut plan). Writes binding-sources.json plus rich
    source witness arrays in binding-source-witnesses.json; the driver may write
    binding-sources.json. Uses no additional helper source files or models.
    """
    import torch

    output = Path(output).resolve(); plan = config["binding_plan"]
    require(plan["schema"] == "mdb-binding-support-complement-next-cut-v1" and [arm["id"] for arm in plan["arms"]] == list(ARMS), "binding plan schema/arms differ")
    require(plan["boundary"]["step"] == 2 and plan["boundary"]["phase"] == "joint" and plan["boundary"]["index"] == 1, "binding plan boundary differs")
    carrier = Evidence(config["binding_source_root"], config["binding_source_manifest_sha256"], config["binding_source_request_sha256"])
    route = Evidence(config["binding_route_root"], config["binding_route_manifest_sha256"], config["binding_route_request_sha256"])
    for evidence in (carrier, route):
        require(not output.is_relative_to(evidence.root) and not evidence.root.is_relative_to(output)
            and not Path(store.root).resolve().is_relative_to(evidence.root)
            and not evidence.root.is_relative_to(Path(store.root).resolve()), "output/store overlaps immutable source")
    require(carrier.manifest_sha256 == plan["source_inventory"], "binding plan/source inventory differs")
    source_config = carrier.request["config"]
    require(source_config["prior_root"] == str(route.root) and source_config["prior_manifest_sha256"] == route.manifest_sha256
        and source_config["prior_request_sha256"] == route.request["request_sha256"], "carrier/route lineage differs")
    require(source_config["contexts"] == plan["contexts"] == route.request["config"]["contexts"], "source context identity differs")
    require(carrier.read("summary.json")["job_id"] == plan["source_native_job"], "carrier native job differs")
    parents = carrier.read("source-parents.json")
    crows = {(row["case"], row["arm"]): row for row in carrier.read("rows.json")}
    rrows = {(row["case"], row["arm"]): row for row in route.read("rows.json")}
    rfields = {(row["case"], row["arm"]): row for row in route.read("fields.json")}
    output.mkdir(parents=True, exist_ok=True); prepared, cases = {}, {}
    for context in plan["contexts"]:
        case = context["id"]; chosen = plan["cases"][case]
        require(Path(case).name == case and case not in ("", ".", ".."), "unsafe case identity")
        require(parents[case] == chosen["parents"], "predeclared source parents differ")
        p, q = (carrier.cut(chosen["parents"][role]) for role in ("P", "Q"))
        for cut in (p, q):
            require_clock(cut, adapter, 1); store.save(cut)
        common = require_pair(p, q); tp, tq = p.payload["text"], q.payload["text"]
        require(tp.ndim == tq.ndim == 3 and tp.shape == tq.shape and tp.shape[0] == 1 and tp.dtype == tq.dtype == torch.bfloat16, "paired text layout differs")
        require(bool(torch.isfinite(tp).all()) and bool(torch.isfinite(tq).all()), "nonfinite paired text")
        support = chosen["S"]; total_rows = tp.shape[1]
        require(isinstance(support, list) and support == sorted(set(support)) and all(type(row) is int and 0 <= row < total_rows for row in support), "invalid predeclared support")
        complement = [row for row in range(total_rows) if row not in support]
        require(support and complement, "nonempty support and complement required")
        selectors = route.request["config"]["selectors"][case]
        require(support == sorted(set(selectors["A"]) | set(selectors["B"])), "support differs from source-sealed A union B")
        inputs, input_records = {}, {}
        for role, arm in (("P", "A@A"), ("Q", "A@B")):
            row, field = rrows[case, arm], rfields[case, arm]
            require(row["marks"]["after_joint0"] == chosen["parents"][role] and field["entry_cut"] == row["marks"]["entry"], "joint0 input/output lineage differs")
            entry = route.cut(field["entry_cut"]); require_clock(entry, adapter, 0); store.save(entry)
            for name in ("image", "text"):
                value = route.array(field["actual_"+name])
                require(same_bits(value, entry.payload[name]), "actual entry array/StateCut slot differs: " + name)
            inputs[role] = entry
            input_records[role] = dict(cut=cut_record(entry), arrays={name: field["actual_"+name] for name in ("image", "text")})
        require_pair(inputs["P"], inputs["Q"])
        pi, qi = inputs["P"].payload["text"], inputs["Q"].payload["text"]
        require(pi.shape == qi.shape == tp.shape and same_bits(pi[:, complement], qi[:, complement]), "joint0 input differs outside original support")
        require(same_bits(inputs["P"].payload["image"], inputs["Q"].payload["image"]), "joint0 input image differs")
        conditioning = q.payload["conditioning"].clone()
        require(same_bits(conditioning, inputs["Q"].payload["conditioning"]), "project3 reset conditioning lineage differs")
        payloads = dict(whole_Q=tq.clone(), support_P=tq.clone(), complement_P=tp.clone(), whole_P=tp.clone())
        payloads["support_P"][:, support] = tp[:, support]
        payloads["complement_P"][:, support] = tq[:, support]
        parallelogram = (tp.double()-payloads["support_P"].double())-(payloads["complement_P"].double()-tq.double())
        require(int(torch.count_nonzero(parallelogram)) == 0, "exact row-replacement parallelogram differs")
        prefix = "binding-source/"+case+"/"
        def witness(name, value):
            return save_tensor(output, prefix+name+".npy", value)
        writes = {role: cut.payload["text"].double()-inputs[role].payload["text"].double() for role, cut in (("P", p), ("Q", q))}
        delta = tp.double()-tq.double(); write_delta = writes["P"]-writes["Q"]
        witnesses = dict(input_text={role: witness("joint0-input-"+role+"-text", cut.payload["text"]) for role, cut in inputs.items()},
            input_image={role: witness("joint0-input-"+role+"-image", cut.payload["image"]) for role, cut in inputs.items()},
            output_text={"P": witness("joint0-output-P-text", tp), "Q": witness("joint0-output-Q-text", tq)},
            output_image={"P": witness("joint0-output-P-image", p.payload["image"]), "Q": witness("joint0-output-Q-image", q.payload["image"])},
            output_text_difference=witness("output-text-P-minus-Q", delta),
            native_finite_writes={role: witness("native-finite-write-"+role, value) for role, value in writes.items()},
            native_finite_write_difference=witness("native-finite-write-P-minus-Q", write_delta),
            entry_parallelogram=witness("entry-parallelogram", parallelogram))
        for role in ("P", "Q"):
            input_records[role]["source_arrays"] = input_records[role]["arrays"]
            input_records[role]["arrays"] = {"image": witnesses["input_image"][role], "text": witnesses["input_text"][role]}
        text_records = {arm: witness("payload-"+arm, value) for arm, value in payloads.items()}
        conditioning_record = witness("project3-base-conditioning", conditioning)
        known = {}
        for arm, historical in (("whole_Q", "I.Q_T.Q"), ("whole_P", "I.Q_T.P")):
            row = crows[case, historical]
            require(row["marks"]["entry"] == chosen["known_endpoints"][arm], "known endpoint entry differs from plan")
            entry = carrier.cut(row["marks"]["entry"]); require_clock(entry, adapter, 1)
            require(slots({name: value for name, value in entry.payload.items() if name != "text"}) == slots({name: value for name, value in q.payload.items() if name != "text"})
                and same_bits(entry.payload["text"], payloads[arm]), "known whole-T entry closure differs")
            final = carrier.cut(row["final_cut"]); require_clock(final, adapter, None, step=4, phase="project"); store.save(final)
            png = carrier.member(row["png"]); require(sha(png) == row["png_sha256"], "known PNG identity differs")
            known[arm] = dict(entry_cut=entry.fingerprint, cut=final.fingerprint, png=row["png"], png_sha256=sha(png), source_root=str(carrier.root))
        analysis = dict(orientation="P-minus-Q", input_text={part: compare(pi, qi, rows) for part, rows in (("support", support), ("complement", complement))},
            output_text={part: compare(tp, tq, rows) for part, rows in (("support", support), ("complement", complement))},
            output_image=compare(p.payload["image"], q.payload["image"]),
            native_finite_write_difference={part: compare(writes["P"], writes["Q"], rows) for part, rows in (("support", support), ("complement", complement))},
            complement_write_vs_output_difference_maxabs=float((write_delta[:, complement]-delta[:, complement]).abs().max()),
            entry_parallelogram=dict(formula="TP - Tsupport - Tcomplement + TQ", arithmetic="FP64 (TP-Tsupport) - (Tcomplement-TQ)", max_abs=float(parallelogram.abs().max()), nonzero=0),
            common_nonstream_slots=common, fixed_nontext_source="Q", raw_input_prediction=False,
            interpretation="Finite native joint0 text-write difference outside source-sealed S where prior input is byte-identical; supplied-state observation, not an identified writer or semantic quality gate.")
        metadata = dict(pcut=cut_record(p), qcut=cut_record(q), joint0_inputs=input_records, support=support, complement=complement,
            text_payloads=text_records, base_conditioning=conditioning_record, witnesses=witnesses, analysis=analysis, known_endpoints=known)
        cases[case] = metadata
        prepared[case] = dict(pcut=p, qcut=q, support=list(support), text_payloads=payloads, base_conditioning=conditioning,
            known_endpoints=known, witnesses=metadata, analysis=analysis)
    provenance = dict(carrier=carrier.receipt(), route=route.receipt())
    for evidence in (carrier, route):
        for name in list(evidence.consumed):
            evidence.member(name)
    save_json(output / "binding-source-witnesses.json", dict(schema=SCHEMA, source_only=True, model_identity=adapter.model_identity,
        binding_plan=plan, cases=cases, provenance=provenance, new_model_calls=0))
    for case in prepared.values():
        case["provenance"] = provenance
    return prepared
