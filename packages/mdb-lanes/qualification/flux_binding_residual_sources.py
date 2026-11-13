"""Saved source custody and source-only residual / finite net-write endpoints."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

SCHEMA = "mdb-flux-binding-residual-sources-v1"
PREVIOUS = "flux_binding_support_sources.py"
PREVIOUS_SHA256 = "217331e363a698babfdd9349fad8b4bc786601eb2d76220a3c733507b26e41a0"
ARMS = ("whole_Q", "residual_only", "write_only", "both_RW")
CAST = "Torch2.13 CPU FP64 Tensor.to(torch.bfloat16), directly; no explicit FP32 intermediate; no ideal correct-rounding promise"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def previous():
    path = Path(__file__).with_name(PREVIOUS)
    require(sha(path) == PREVIOUS_SHA256, "frozen source helper differs")
    return runpy.run_path(str(path))


def decompose_sources(pin, qin, pout, qout, support):
    """Use saved BF16 values only; retain cast, landing and finite remainders."""
    import torch

    arrays = dict(Pin=pin, Qin=qin, Pout=pout, Qout=qout)
    require(pin.ndim == 3 and pin.shape[0] == 1, "batch-one text required")
    require(all(value.shape == pin.shape and value.dtype == torch.bfloat16 and value.device.type == "cpu"
                and bool(torch.isfinite(value).all()) for value in arrays.values()), "finite paired CPU BF16 text required")
    require(isinstance(support, list) and support == sorted(set(support)) and support
            and all(type(row) is int and 0 <= row < pin.shape[1] for row in support), "invalid source support")
    complement = [row for row in range(pin.shape[1]) if row not in support]
    require(complement, "source support must have a complement")
    wide = {name: value.double() for name, value in arrays.items()}
    wp, wq = wide["Pout"] - wide["Pin"], wide["Qout"] - wide["Qin"]
    dr = wide["Pin"] - wide["Qin"]
    dw = wp - wq
    total = dr + dw
    output_difference = wide["Pout"] - wide["Qout"]
    closure = total - output_difference
    require(int(torch.count_nonzero(closure)) == 0, "FP64 residual/write decomposition does not close")
    masked = {}
    for name, value in dict(DR=dr, DW=dw, DR_plus_DW=total).items():
        masked[name] = torch.zeros_like(value)
        masked[name][:, support] = value[:, support]
    ideal = {"whole_Q": wide["Qout"].clone()}
    payloads = {"whole_Q": qout.clone()}
    for arm, delta in zip(ARMS[1:], (dr, dw, total), strict=True):
        ideal[arm] = wide["Qout"].clone()
        ideal[arm][:, support] = wide["Qout"][:, support] + delta[:, support]
        payloads[arm] = qout.clone()
        payloads[arm][:, support] = ideal[arm][:, support].to(torch.bfloat16)
    helper = previous()
    expected = qout.clone()
    expected[:, support] = pout[:, support]
    require(helper["same_bits"](payloads["both_RW"], expected), "combined endpoint differs from exact support_P source")
    require(all(helper["same_bits"](value[:, complement], qout[:, complement]) for value in payloads.values()),
            "endpoint changed source complement bits")
    landed = {arm: value.double() - wide["Qout"] for arm, value in payloads.items()}
    rounding = {arm: payloads[arm].double() - ideal[arm] for arm in ARMS}
    remainder = (payloads["both_RW"].double() - payloads["residual_only"].double()) - (
        payloads["write_only"].double() - payloads["whole_Q"].double())
    equivalent = dict(residual_only=wide["Pin"] + wq, write_only=wide["Qin"] + wp, both_RW=wide["Pin"] + wp)
    equivalent_remainders = {arm: ideal[arm][:, support] - value[:, support] for arm, value in equivalent.items()}
    r, w = dr[:, support].reshape(-1), dw[:, support].reshape(-1)
    rn, wn = float(r.norm()), float(w.norm())
    dot = float(torch.dot(r, w))
    analysis = dict(
        orientation="P-minus-Q", definition="Net finite joint0 output-minus-input includes attention, MLP and finite additions; no MLP-only claim",
        support=support, complement=complement, cast_operator=CAST,
        endpoint_parenthesization=dict(residual_only="Qout + DR", write_only="Qout + DW", both_RW="Qout + (DR + DW)"),
        geometry=dict(residual_l2=rn, write_l2=wn, dot=dot, cosine=dot / (rn * wn) if rn and wn else None,
                      combined_l2=float(total[:, support].norm())),
        closure=helper["compare"](total, output_difference),
        cast_landing={arm: helper["compare"](payloads[arm].double(), ideal[arm]) for arm in ARMS},
        entry_nonadditive_remainder=helper["compare"](remainder, torch.zeros_like(remainder)),
        equivalent_wide_corner_max_abs={arm: float(value.abs().max()) for arm, value in equivalent_remainders.items()},
        source_support_control_bit_exact=True, outside_support_bit_exact=True,
    )
    return dict(source_arrays=arrays, widened=wide, finite_writes=dict(P=wp, Q=wq),
                deltas=dict(DR=dr, DW=dw, DR_plus_DW=total, output_difference=output_difference, closure=closure),
                masked_deltas=masked, ideal_endpoints=ideal, text_payloads=payloads,
                realized_displacements=landed, cast_errors=rounding,
                entry_nonadditive_remainder=remainder, analysis=analysis)


class Evidence:
    """Authenticate consumed lifecycle members and complete consumed public CAS."""
    def __init__(self, source):
        self.source = dict(source)
        self.root = Path(source["root"]).resolve()
        require(sha(self.root / "files.json") == source["manifest_sha256"], "lifecycle inventory differs")
        self.manifest = json.loads((self.root / "files.json").read_text())
        require(self.manifest["job_id"] == source["job_id"] and self.manifest["request_sha256"] == source["request_sha256"], "lifecycle wrapper identity differs")
        self.files, self.consumed = self.manifest["files"], {}
        request = self.read("request.json")
        identity = hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                            sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(identity == request["request_sha256"] == source["request_sha256"], "lifecycle request differs")
        require(request["source_hashes"].get("flux_lifecycle_campaign.py") == "af2eeaf0ee069c57c911cf53cbd12f12a637c5f2d226f16a2d507a9521a952c1", "lifecycle producer differs")
        require(self.read("plan.json") == dict(request["config"], job_id=source["job_id"],
                request_sha256=source["request_sha256"], output=str(self.root)), "lifecycle plan/request differs")
        self.request = request

    def member(self, relative):
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe source member")
        path = self.root / relative
        require(path.resolve().is_relative_to(self.root) and not any(item.is_symlink() for item in (path, *path.parents)), "escaped source member")
        require(relative in self.files and path.is_file(), "uninventoried source member")
        record = self.files[relative]
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "lifecycle member changed: " + relative)
        self.consumed[relative] = dict(record)
        return path

    def read(self, relative):
        return json.loads(self.member(relative).read_text())

    def cut(self, record, adapter, store, helper):
        from saturn_pub.store import LocalStore

        fingerprint = record["cut"]
        require(len(fingerprint) == 64 and all(value in "0123456789abcdef" for value in fingerprint), "invalid source cut")
        metadata = self.read("state-store/cuts/" + fingerprint + ".json")
        for page in metadata["slots"].values():
            self.member("state-store/pages/" + page + ".json")
        for blob in metadata["physical_blobs"]:
            self.member("state-store/blobs/" + blob + ".safetensors")
        cut = LocalStore(self.root / "state-store").load(fingerprint, device="cpu")
        require(cut.fingerprint == fingerprint and cut.model_identity == adapter.model_identity, "source strict model identity differs")
        actual = helper["cut_record"](cut)
        for key in ("parent", "model_identity", "execution", "boundary", "payload_descriptors"):
            if key in record:
                require(actual[key] == record[key], "source cut descriptor differs: " + key)
        if "point" in record or "execution_point" in record:
            require(actual["point"] == record.get("point", record.get("execution_point")), "source full point differs")
        store.save(cut)
        return cut

    def receipt(self):
        return dict(self.source, verification_scope="Consumed inventoried members and complete consumed StateCut closures and forecast transaction",
                    consumed_members=dict(sorted(self.consumed.items())))


def prepare_sources(config, output, store, adapter, base, old):
    """Reuse already authenticated/copied lifecycle observations, never a writer."""
    from mdb_lanes.forecast_transaction import verify_transaction

    output = Path(output).resolve()
    evidence, helper = Evidence(config["source"]), previous()
    require(not output.is_relative_to(evidence.root) and not evidence.root.is_relative_to(output)
            and not Path(store.root).resolve().is_relative_to(evidence.root), "output overlaps immutable source")
    summary = evidence.read("summary.json")
    require(summary["job_id"] == config["source"]["job_id"] and summary["request_sha256"] == config["source"]["request_sha256"], "source summary identity differs")
    identity_source = evidence.read("source.json")
    require(identity_source["job_id"] == config["source"]["job_id"]
            and identity_source["request_sha256"] == config["source"]["request_sha256"], "source parameter provenance differs")
    for key in old["_source_keys"]():
        require(config[key + "_sha256"] == evidence.request["config"][key + "_sha256"], "parameter/runtime source lineage differs: " + key)
    for relative in evidence.files:
        if relative.startswith("forecast-transaction/"):
            evidence.member(relative)
    verified = verify_transaction(evidence.root / "forecast-transaction", expected_seal_sha256=summary["forecast_seal"])
    source_cases, index = evidence.read("binding-sources.json"), evidence.read("state-index.json")
    cases, prepared = {}, {}
    for cid in config["contexts"]:
        source = source_cases[cid]
        witness = source["witnesses"]
        p, q = (evidence.cut(witness[name], adapter, store, helper) for name in ("pcut", "qcut"))
        helper["require_pair"](p, q)
        for cut in (p, q):
            helper["require_clock"](cut, adapter, 1)
        require(source["P"]["cut"] == p.fingerprint and source["Q"]["cut"] == q.fingerprint, "source P/Q joins differ")
        inputs = {role: evidence.cut(witness["joint0_inputs"][role]["cut"], adapter, store, helper) for role in ("P", "Q")}
        helper["require_pair"](inputs["P"], inputs["Q"])
        for cut in inputs.values():
            helper["require_clock"](cut, adapter, 0)

        def array(record):
            evidence.member(record["path"])
            value = base["load_tensor"](evidence.root, record, "cpu")
            require(list(value.stride()) == record["stride"], "source witness stride differs")
            return value

        arrays = dict(Pin=array(witness["joint0_inputs"]["P"]["arrays"]["text"]),
                      Qin=array(witness["joint0_inputs"]["Q"]["arrays"]["text"]),
                      Pout=array(witness["witnesses"]["output_text"]["P"]),
                      Qout=array(witness["witnesses"]["output_text"]["Q"]))
        for name, value in arrays.items():
            cut = inputs[name[0]] if name.endswith("in") else {"P": p, "Q": q}[name[0]]
            require(helper["same_bits"](value, cut.payload["text"]), "saved source text/cut differs")
        support = source["support"]
        require(support == witness["support"] == config["binding_plan"]["cases"][cid]["S"], "original source support changed")
        math = decompose_sources(**{name.lower(): value for name, value in arrays.items()}, support=support)
        complement = [row for row in range(q.payload["text"].shape[1]) if row not in support]
        require(helper["same_bits"](arrays["Pin"][:, complement], arrays["Qin"][:, complement])
                and helper["same_bits"](inputs["P"].payload["image"], inputs["Q"].payload["image"]), "original joint0 input outside support differs")
        for arm, old_arm in (("whole_Q", "whole_Q"), ("both_RW", "support_P")):
            require(helper["same_bits"](math["text_payloads"][arm], array(source["text_payloads"][old_arm])), "known endpoint payload differs")
        conditioning = array(source["base_conditioning"])
        require(helper["same_bits"](conditioning, q.payload["conditioning"]), "base reset conditioning differs")
        prefix = "residual-write-source/" + cid + "/"

        def write(name, value):
            return old["_save_tensor"](output, prefix + name + ".npy", value)

        records = {group: {name: write(group + "/" + name, value) for name, value in math[group].items()}
                   for group in ("source_arrays", "widened", "finite_writes", "deltas", "masked_deltas", "ideal_endpoints", "text_payloads", "realized_displacements", "cast_errors")}
        records["entry_nonadditive_remainder"] = write("entry-nonadditive-remainder", math["entry_nonadditive_remainder"])
        known = {}
        for arm, old_arm in (("whole_Q", "whole_Q"), ("both_RW", "support_P")):
            path = index["binding-" + cid + "/" + old_arm]
            rows = evidence.read(path["root"] + "/native-rows.json")
            entry = evidence.cut(rows[0]["retained"], adapter, store, helper)
            require(helper["same_bits"](entry.payload["text"], math["text_payloads"][arm])
                    and helper["slots"]({name: value for name, value in entry.payload.items() if name != "text"})
                    == helper["slots"]({name: value for name, value in q.payload.items() if name != "text"}), "known native entry closure differs")
            final = evidence.cut(dict(cut=path["native_final"]), adapter, store, helper)
            helper["require_clock"](final, adapter, None, step=4, phase="project")
            decoder = evidence.read(path["root"] + "/decoder-native-reference.json")
            png = path["root"] + "/" + decoder["png"]["path"]
            require(sha(evidence.member(png)) == decoder["png"]["sha256"], "known PNG differs")
            known[arm] = dict(source_root=str(evidence.root), source_manifest_sha256=config["source"]["manifest_sha256"],
                              entry_cut=entry.fingerprint, cut=final.fingerprint, png=png, png_sha256=sha(evidence.root / png), historical_arm=old_arm)
        case = dict(P=helper["cut_record"](p), Q=helper["cut_record"](q),
                    joint0_inputs={role: helper["cut_record"](cut) for role, cut in inputs.items()},
                    support=support, complement=complement, **records, base_conditioning=write("base-conditioning", conditioning),
                    analysis=math["analysis"], known_endpoints=known)
        cases[cid] = case
        prepared[cid] = dict(qcut=q, pcut=p, text_payloads=math["text_payloads"], base_conditioning=conditioning, metadata=case)
    for relative in tuple(evidence.consumed):
        evidence.member(relative)
    sidecar = dict(schema=SCHEMA, source=config["source"], source_only=True, cases=cases,
                   provenance=evidence.receipt(), source_transaction=verified, new_model_calls=0)
    helper["save_json"](output / "residual-write-sources.json", sidecar)
    return prepared, sidecar
