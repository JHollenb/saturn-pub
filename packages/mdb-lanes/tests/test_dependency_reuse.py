"""Footprint planning keeps declaration authority separate from execution."""
import copy
import unittest

from mdb_lanes.dependency_reuse import DependencyReuseError, plan_suffix_reuse, verify_reuse_plan


def plan(steps, changed=("latent",)):
    return plan_suffix_reuse(steps, changed_slots=changed, model_identity="model",
        execution_identity="deterministic-contract", source_parent="source", target_parent="target")


def transition(operation, reads, writes, footprint="declared"):
    return {"operation": operation, "reads": reads, "writes": writes, "footprint": footprint}


class DependencyReuseTests(unittest.TestCase):
    def test_carried_state_waits_for_scheduler(self):
        steps = [transition("block", ["image", "text"], ["image", "text"]),
                 transition("readout", ["image"], ["velocity"]),
                 transition("scheduler", ["latent", "velocity"], ["latent"]),
                 transition("decode", ["latent"], ["rgb"])]
        p = plan(steps)
        self.assertEqual([r["decision"] for r in p["rows"]],
            ["reuse-candidate", "reuse-candidate", "recompute-candidate", "recompute-candidate"])
        self.assertTrue(verify_reuse_plan(p)["verified"])

    def test_identical_input_overwrite_clears_difference(self):
        p = plan([transition("constant", [], ["x"]), transition("reader", ["x"], ["y"])], ("x",))
        self.assertEqual(p["reuse_candidates"], 2)
        self.assertEqual(p["rows"][-1]["changed_after"], [])

    def test_unknown_footprint_taints_later_candidates(self):
        p = plan([transition("opaque", [], [], "unknown"), transition("otherwise-independent", [], ["rgb"])])
        self.assertEqual(p["reuse_candidates"], 0)

    def test_rejects_tampering_and_duplicate_slots(self):
        p = plan([transition("read", ["latent"], ["rgb"])])
        p["rows"][0]["decision"] = "reuse-candidate"
        with self.assertRaises(DependencyReuseError): verify_reuse_plan(p)
        with self.assertRaises(DependencyReuseError): plan([], ("x", "x"))

    def test_inputs_are_owned_and_bound(self):
        steps = [transition("read", ["image"], ["velocity"])]
        original = copy.deepcopy(steps);p = plan(steps)
        steps[0]["reads"].append("latent")
        self.assertEqual(p["rows"][0]["transition"], original[0])
        self.assertEqual(p["authority"], "diagnostic-planning-only")


if __name__ == "__main__": unittest.main()
