"""Small independent arithmetic/identity checks; no trained model execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "qualification/flux_binding_generalization_audit.py"
spec = importlib.util.spec_from_file_location("binding_generalization_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditMechanicsTests(unittest.TestCase):
    def test_ties_to_even_and_zero_sign_are_explicit(self):
        values = np.array([1 + 2**-8, 1 + 3*2**-8, 0.0, -0.0], np.float32)
        actual = audit.round_bf16(values)
        np.testing.assert_array_equal(actual, np.array([0x3f80, 0x3f82, 0, 0x8000], np.uint16))
        self.assertFalse(audit.comparison(actual[2:3], actual[3:4])["raw_bits_exact"])
        self.assertEqual(audit.comparison(actual[2:3], actual[3:4])["rms_error"], 0)

    def test_writer_keeps_both_rounded_residual_additions(self):
        # Each half-ulp attention/MLP write rounds away separately. Combining
        # the two in a real-valued or reassociated sum would advance one ulp.
        bits = lambda x: audit.round_bf16(np.array([[x]], np.float32))
        actual = audit.sequential_writer(bits(1), bits(2**-8), bits(2**-8), bits(1), bits(1))
        np.testing.assert_array_equal(actual, bits(1))
        self.assertFalse(np.array_equal(actual, bits(1 + 2**-7)))

    def test_bad_writer_broadcast_and_nonfinite_refuse(self):
        x = audit.round_bf16(np.ones((1, 3, 2), np.float32))
        with self.assertRaises(audit.AuditError):
            audit.sequential_writer(x, x, x, np.ones((1, 4, 2), np.uint16), x)
        with self.assertRaises(audit.AuditError):
            audit.round_bf16(np.array([float("nan")], np.float32))

    def test_image_projection_zero_and_halves_are_diagnostics(self):
        base = np.zeros((2, 4, 3), np.uint8)
        source = base.copy()
        source[:, :2] = 10
        image = base.copy()
        image[:, :2] = 5
        result = audit.rgb_diagnostics(image, base, source)
        self.assertEqual(result["whole"]["signed_projection_on_natural_source_minus_base"], .5)
        self.assertEqual(result["left"]["response_rms_0_255"], 5)
        self.assertIsNone(result["right"]["signed_projection_on_natural_source_minus_base"])

    def test_inventory_refuses_escaping_paths_and_omitted_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "value.txt").write_text("same input")
            (root / "files.json").write_text('{"value.txt":"' + audit.sha(root / "value.txt") + '"}')
            hashes, result = audit.inventory(root)
            self.assertEqual(result["files"], 1)
            self.assertIn("value.txt", hashes)
            with self.assertRaises(audit.AuditError):
                audit.safe_path(root, "../outside")
            (root / "unlisted.txt").write_text("not sealed")
            with self.assertRaises(audit.AuditError):
                audit.inventory(root)

    def test_duplicate_tagged_keys_refuse(self):
        key = {"kind": "scalar", "value": "x"}
        value = {"kind": "scalar", "value": 1}
        with self.assertRaises(audit.AuditError):
            audit.untag({"kind": "mapping", "items": [{"key": key, "value": value}] * 2})


if __name__ == "__main__":
    unittest.main()
