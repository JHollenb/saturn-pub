"""Explicit worker environment intent and receipts, independent of checkouts."""
import importlib.metadata
import os
import platform
import subprocess

REQUIREMENTS = ("torch==2.13.0", "diffusers==0.39.0", "transformers==5.14.1", "accelerate==1.14.0",
                "safetensors>=0.5", "numpy>=2.0", "pyyaml>=6.0", "psutil>=5.9", "pillow>=10.0")


def intent():
    return {"uv_cache_dir": os.environ.get("MDB_WORKER_UV_CACHE_DIR", "/mnt/big/llm-models/.uv-cache"),
            "torch_home": os.environ.get("MDB_WORKER_TORCH_HOME", "/mnt/big/llm-models/.torch-cache"),
            "extra_model_roots": "/mnt/big/llm-models", "python": "3.14.6",
            "requirements": list(REQUIREMENTS)}


def receipt():
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()}
    def command(args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RuntimeError("environment identity command failed: " + args[0])
        return result.stdout.strip()
    return {"python": platform.python_version(), "packages": packages,
            "uv": command(["uv", "--version"]),
            "cuda_driver": command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
            "resolved": {key: os.environ.get(key) for key in ("UV_CACHE_DIR", "TORCH_HOME", "LLM_MODELS_EXTRA_ROOT")}}
