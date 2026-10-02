"""Keep the circuit custody verifier independent of tensor execution imports."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "src/mdb/bench_circuit_measurement.py").read_text()
tree = ast.parse(source)
functions = [ast.get_source_segment(source, node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in {"_verify_act", "verify_artifact"}]
header = '''"""Tensor-free circuit measurement custody verification."""
from .model_os_contracts import content_digest
SCHEMA = "saturn-bench-circuit-measurement-v1"
OBSERVATION_SCHEMA = "saturn-bench-circuit-observation-v1"

'''
(root / "src/mdb/bench_circuit_verify.py").write_text(header + "\n\n".join(functions) + "\n")
