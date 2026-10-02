"""Package inventory; resident-specific capability catalogs remain authoritative."""


def inventory():
    return {"schema": "mdb-capability-inventory-v1", "runtime": "0.1.0", "families": ["qwen2", "mamba", "flux", "sdxl"],
        "session": ["from_resident", "resume", "prefill", "prepare", "step", "generate", "inspect", "stage_input", "checkpoint", "fork", "checkout", "restore", "replay", "compare", "export", "close"],
        "memory": ["shared-ancestors", "private-continuations", "copy-on-write", "durable-pages", "residency-admission", "eviction", "hydration", "ownership-accounting", "four-plane-bundles"],
        "debugger": ["safe-point-breakpoints", "watchpoints", "typed-console", "source-frames", "preview", "commit", "abort", "checkpoint-navigation", "mrun-transport"],
        "research": ["hyperion-bindings", "discover", "replay_observation", "measure_circuit", "replay_circuit"],
        "history": ["thott", "future-forest", "program-recorder", "retained-rejected-futures"],
        "evidence": ["bisect", "xref", "claims", "watch", "custody", "cite", "run", "verify"],
        "authority": "Call session.capabilities() for the resident's actual stages, writes, closure and numerical contract.",
        "limits": ["single writer and serialized execution", "exactness is family/checkpoint/toolchain/lane bound", "Qwen KV interventions are read-only observations", "circuit measurement is Qwen/Mamba journal-only", "deployed broker qualification requires separately available mrun routes", "no integrated CUDA VMM or universal semantic reasoning detector"]}
