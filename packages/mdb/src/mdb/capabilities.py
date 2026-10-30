"""Package inventory; resident-specific capability catalogs remain authoritative."""


def inventory():
    from .backends import available_backends
    return {"schema": "mdb-capability-inventory-v1", "runtime": "0.1.0", "families": ["qwen2", "mamba", "flux", "sdxl"],
        "local_debugger": {"entry": "mdb debug", "base_demo": "pure-Python Counter (in-memory)",
            "native_demo": "mdb debug --tiny qwen2", "checkpoint_loader": ["qwen2", "mamba"],
            "additional_demos": ["mdb debug --tiny transformerlens"],
            "front_doors": ["debug", "open --local", "resume --local"],
            "output": ["compact frame", "--full", "--json"], "requires_mrun": False},
        "backend_extensions": {"constructor": "BenchSession.from_backend", "factories": list(available_backends()),
            "interface": "ModelRunner", "forward": ["capture", "interventions"],
            "stateful_interface": "DebuggerAdapter", "stateful_support": "backend-declared",
            "activation_capture_is_checkpoint": False},
        "transformerlens_debugger": {"adapter": "TransformerLensDebuggerAdapter",
            "contract": "transformerlens-cpu-cachefree-layer-reentry-v1",
            "families": ["native TL", "GPT-2", "Qwen2/Qwen2.5"],
            "stateful": ["layer stepping", "break/watch", "residual edits", "hook specs", "fork", "export/resume"],
            "kv_cache": False, "incremental_generation": False},
        "session": ["from_resident", "resume", "prefill", "prepare", "step", "generate", "inspect", "stage_input", "checkpoint", "fork", "checkout", "restore", "replay", "compare", "export", "close"],
        "memory": ["shared-ancestors", "private-continuations", "copy-on-write", "durable-pages", "residency-admission", "eviction", "hydration", "ownership-accounting", "four-plane-bundles"],
        "debugger": ["safe-point-breakpoints", "watchpoints", "typed-console", "source-frames", "preview", "commit", "abort", "checkpoint-navigation", "mrun-transport"],
        "research": ["hyperion-bindings", "discover", "replay_observation", "measure_circuit", "replay_circuit"],
        "computation_forecasts": {
            "prefix": "mdb.forward_prefix.record_prefix; CPU eval, named entry, occurrence zero",
            "ports": "record_forward(capture_ports=True); actual module arguments/returns and named state roots",
            "transaction": "ForecastTransaction.run_script; trusted Python callbacks, durable seal before release",
            "comparisons": "compare_equations; signed residuals, declared decisions, explicit target/collateral baseline",
            "catalog": "EquationCatalog; immutable source-bound records and supplied applicability predicates",
            "front_doors": ["mdb ports", "mdb forecast", "mdb equations"],
            "limits": ["callbacks are not an OS sandbox", "applicability matching does not establish generalization",
                       "native CPU contract requires model-specific qualification"]},
        "history": ["thott", "future-forest", "program-recorder", "retained-rejected-futures"],
        "evidence": ["bisect", "xref", "claims", "watch", "custody", "cite", "run", "verify"],
        "authority": "Call session.capabilities() for the resident's actual stages, writes, closure and numerical contract.",
        "limits": ["single writer and serialized execution", "exactness is family/checkpoint/toolchain/lane bound", "Qwen KV reads and bounded row/depth/component writes require journal mode and committed token boundaries", "circuit measurement is Qwen/Mamba journal-only", "deployed broker qualification requires separately available mrun routes", "no integrated CUDA VMM or universal semantic reasoning detector"]}
