"""The production decoder's largest saved tensor also needs NPY framing."""
import io
import runpy
from pathlib import Path

import numpy as np


def test_original_macro_geometry_fits_with_real_numpy_header_without_allocation():
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "qualification/stage_flux_raw_to_image.py"))
    budget = namespace["decoder_output_budget"]({"block_out_channels": [128, 256, 512, 512]}, 512, 512)
    shape = (1, 256, 512, 512)
    payload_bytes = int(np.prod(shape)) * np.dtype("uint16").itemsize
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<u2", "fortran_order": False, "shape": shape})
    framed_bytes = len(header.getvalue()) + payload_bytes
    assert payload_bytes == 128 << 20
    assert len(header.getvalue()) == 128
    assert framed_bytes > 128 << 20
    assert budget["largest_tensor_payload_bytes"] == payload_bytes
    assert budget["max_member_bytes"] >= framed_bytes
