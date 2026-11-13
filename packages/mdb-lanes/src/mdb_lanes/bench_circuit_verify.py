"""Tensor-free circuit measurement custody verification."""
from .model_os_contracts import content_digest

SCHEMA = "saturn-bench-circuit-measurement-v1"
OBSERVATION_SCHEMA = "saturn-bench-circuit-observation-v1"

def _verify_act(act, artifacts):
    cut = artifacts.get(act["cut_id"])
    receipt = artifacts.get(act["receipt_ref"])
    body = dict(receipt); digest = body.pop("content_sha256")
    if content_digest(body) != digest or receipt["decision"] != "commit":
        raise ValueError("circuit Act receipt is not a valid commit")
    if cut.get("receipt_ref") and cut["receipt_ref"] != act["receipt_ref"]:
        raise ValueError("circuit Act cut/receipt mismatch")
    if act.get("donor_cut"):
        if "pending_token_policy" in act:
            parent = artifacts.get(act["parent_cut"])
            donor = artifacts.get(act["donor_cut"])
            expected_history = {"length": len(parent["token_ids"]),
                                "token_ids_sha256": content_digest(parent["token_ids"])}
            expected_pending = {"recipient_token_id": parent["pending"],
                                "donor_token_id": donor["pending"],
                                "policy": act["pending_token_policy"]}
            expected_act_id = content_digest({
                "parent_cut": act["parent_cut"], "donor_cut": act["donor_cut"],
                "model_abi": content_digest(parent["identity"]),
                "native_abi": parent["numerical_contract"], "history": parent["token_ids"],
                "pending": expected_pending, "geometry": act["geometry"],
                "selector": act["selector"], "pending_token_policy": act["pending_token_policy"],
                "selected_payload": act["selected_payload"]})
            if (donor["identity"] != parent["identity"] or donor["lane"] != parent["lane"]
                    or donor["numerical_contract"] != parent["numerical_contract"]
                    or donor["token_ids"] != parent["token_ids"]
                    or donor["prompt_length"] != parent["prompt_length"]):
                raise ValueError("circuit successor KV donor/history mismatch")
            if (act.get("history") != expected_history or act.get("pending") != expected_pending
                    or act.get("act_id") != expected_act_id):
                raise ValueError("circuit successor KV Act binding mismatch")
            if (cut["parent_cut"] != act["parent_cut"] or cut["identity"] != parent["identity"]
                    or cut["lane"] != parent["lane"]
                    or cut["numerical_contract"] != parent["numerical_contract"]
                    or cut["token_ids"] != parent["token_ids"]
                    or cut["prompt_length"] != parent["prompt_length"]
                    or cut["pending"] != parent["pending"] or not cut["intervention_lineage"]
                    or cut["intervention_lineage"][-1] != act["act_id"]):
                raise ValueError("circuit successor KV committed cut mismatch")
            continuation, backend = receipt["continuation"], receipt["backend"]
            if (receipt.get("schema") != "saturn-model-os-execution-receipt-v1"
                    or receipt.get("consumer_id") != "bench.qwen.successor-kv-install"
                    or continuation.get("accepted") is not True
                    or continuation.get("consumer_id") != receipt["consumer_id"]
                    or continuation.get("horizon") != 0
                    or continuation.get("execution_mode") != "native-successor-kv-install"
                    or continuation.get("act_id") != act["act_id"]
                    or continuation.get("donor_cut") != act["donor_cut"]
                    or continuation.get("recipient_policy") != act["pending_token_policy"]
                    or continuation.get("geometry") != act["geometry"]
                    or continuation.get("selector") != act["selector"]
                    or continuation.get("locality") != act["locality"]
                    or backend.get("execution_mode") != "native-successor-kv-install"
                    or backend.get("locality") != act["locality"]
                    or not isinstance(backend.get("native_statecut"), dict)):
                raise ValueError("circuit successor KV receipt closure mismatch")
        else:
            output = receipt["continuation"]["output"]
            if output["act_id"] != act["act_id"] or output["donor_cut"] != act["donor_cut"]:
                raise ValueError("circuit memory Act donor/receipt mismatch")

def verify_artifact(value, artifacts, manifest, vm_artifacts=None):
    """Verify measurement cut/evidence references without executing a model."""
    if not isinstance(value, dict) or value.get("schema") not in {SCHEMA, OBSERVATION_SCHEMA}: return False
    if value["model_identity"] != manifest["identity"] or value["lane"] != manifest["lane"]:
        raise ValueError("circuit evidence model/lane identity differs from manifest")
    if content_digest(value["spec"]) != value["spec_sha256"]: raise ValueError("circuit spec digest mismatch")
    for key in ("parent_cut", "site_cut", "replay_cut", "donor_cut", "recipient_cut", "symbol_ref"):
        if value.get(key): artifacts.get(value[key])
    if value["schema"] == SCHEMA:
        for row in value["rows"]:
            recorded = artifacts.get(row["evidence_ref"])
            if recorded != {k: v for k, v in row.items() if k != "evidence_ref"}:
                raise ValueError("circuit report row differs from frozen observation")
            verify_artifact(recorded, artifacts, manifest, vm_artifacts)
    else:
        for key in ("first", "future", "reference_first", "reference_future"):
            observation = value[key]
            if observation is None: continue
            artifacts.get(observation.get("successor_cut", observation.get("endpoint_cut")))
            closure = dict(observation["closure"]); digest = closure.pop("sha256")
            if content_digest(closure) != digest: raise ValueError("circuit closure digest mismatch")
        for key in ("act", "discarded_act"):
            if value.get(key): _verify_act(value[key], artifacts)
        def refs(node):
            if isinstance(node, dict):
                for key, child in node.items():
                    if key in {"cut", "cut_id", "parent_cut", "site_cut", "replay_cut", "donor_cut",
                               "recipient_cut", "successor_cut", "endpoint_cut", "baseline_cut"} and isinstance(child, str):
                        referenced = artifacts.get(child)
                        if referenced["identity"] != manifest["identity"] or referenced["lane"] != manifest["lane"]:
                            raise ValueError("circuit nested cut identity mismatch")
                    elif key == "receipt_ref" and isinstance(child, str):
                        try: artifacts.get(child)
                        except Exception as error:
                            from .circuit_runtime import CircuitRuntimeError
                            if not isinstance(error, CircuitRuntimeError) or not isinstance(error.__cause__, FileNotFoundError): raise
                            if vm_artifacts is None: raise
                            vm_artifacts.get(child)
                    else: refs(child)
            elif isinstance(node, list):
                for child in node: refs(child)
        refs(value)
    return True
