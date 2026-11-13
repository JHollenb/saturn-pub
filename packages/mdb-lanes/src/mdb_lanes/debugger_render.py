"""Bounded human views over unchanged debugger protocol results."""

import json

GROUP_HELP = """MDB debugger command groups (available before attaching):
  Motion:  step [KIND] [COUNT], continue, pause       example: step 1
  Breaks:  break KIND INDEX, watch PATH PREDICATE    example: break step 5
  State:   checkpoint NAME, restore NAME, fork NAME example: checkpoint parent
  Edits:   set PATH VALUE, preview, commit, abort    example: set value 10
  Memory:  inspect PATH, memory read|write|diff|probe example: inspect memory
  History: cuts, reverse-step, reverse-continue      example: reverse-continue value < 3
  Output:  inspect [PATH] --full | --json; help; quit

Start: mdb debug (pure-Python Counter), mdb debug --tiny qwen2|transformerlens,
       mdb debug --local CHECKPOINT_DIR, or mdb debug JOB_ID.
Native Qwen uses layer/token boundaries and residual edits; the Counter uses step/value.
"""


def short_ref(value):
    return str(value).rsplit("/", 1)[-1][:10] if value else "-"


def frame(status):
    runtime = status.get("runtime", status)
    cursor = runtime.get("cursor") or {}
    if not isinstance(cursor, dict):
        cursor = {"step": cursor}
    fields = [f"phase={cursor.get('phase', 'unknown')}"]
    if cursor.get("next_layer") is not None:
        fields.append(f"next_layer={cursor['next_layer']}")
    if runtime.get("token_count") is not None:
        fields.append(f"tokens={runtime['token_count']}")
    elif "token" in cursor:
        fields.append(f"generated={cursor['token']}")
    for key, label in (("step", "step"), ("pending_token", "pending")):
        if cursor.get(key) is not None:
            fields.append(f"{label}={cursor[key]}")
    if "value" in runtime:
        fields.append(f"value={runtime['value']}")
    fields.append("cut=" + short_ref(runtime.get("cut_id")))
    if status.get("pending_act"):
        fields.append("edit=pending")
    reason = status.get("pause_reason") or {}
    if reason.get("breakpoints"):
        fields.append("stop=breakpoint")
    elif reason.get("watches"):
        fields.append("stop=watch")
    text = f"({runtime.get('branch', 'main')}) " + " ".join(fields)
    focus = runtime.get("circuit_focus")
    if focus:
        text += (
            f"\nFocus: {focus['component']} {short_ref(focus['record_id'])}"
            f" target={focus['consumer']}"
            + (" [historical occurrence]" if focus["historical"] else "")
        )
    cot = runtime.get("cot") or {}
    if cot.get("enabled"):
        entries = cot.get("entries") or []
        if entries:
            entry = entries[-1]
            terms = ", ".join(repr(item["text"]) for item in entry["candidates"][:5])
            text += (
                f"\nCoT register: token {entry['token_position']}, "
                f"after layer {entry['completed_layer_boundary']} [decoded candidates]"
                f"\n  Vocabulary readout: {terms}"
                "\n  Dependency has not been causally tested."
            )
        else:
            text += "\nCoT register: waiting for a configured layer boundary."
        if entries and not cot.get("current_boundary_observed"):
            text += "\n  Historical observation; current boundary has no matching readout."
        if cot.get("errors"):
            text += "\n  Observer error: " + cot["errors"][-1]["error"]
        if cot.get("panel_error"):
            text += "\n  Graph pane unavailable: " + cot["panel_error"]
    return text


def summary(value, *, depth=0):
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    if isinstance(value, dict):
        if "focus" in value and "state_materialized" in value:
            selected = value["focus"]
            return (
                f"Selected: {selected['component']} node={short_ref(selected['node_id'])}\n"
                f"Target: {selected['consumer']}\n"
                f"Evidence: {value['authority']}; {len(value['frontier'])} frontiers; "
                f"{len(value['results'])} saved results\n"
                "Next: project seek | project explain --view mlp|attention|effect|frequency | project compare"
            )
        if value.get("schema") == "mdb-circuit-investigation-v1":
            head = f"{value['kind']} {value['record_id']}"
            if value["kind"] == "focus":
                return (
                    head
                    + f"\nSelected: {value['component']} {value['boundary']}\nTarget: {value['consumer']}"
                )
            if value["kind"] == "comparison-plan":
                return (
                    head
                    + f"\n{len(value['edits'])} edits; {value['steps']} transition budget; tokens={value['tokens']}\nNext: project compare --run"
                )
            if value["kind"] == "comparison":
                return (
                    head
                    + f"\nSigned effects: {value['effects']}\nInteraction: {value['interaction']}\nParent preserved; terminal status: {value['terminal_status']}"
                )
            if value["kind"] == "materialization":
                return (
                    head
                    + f"\n{value['execution_point']}\nValue: {summary(value['value'])}\nSample: {value['sample']}"
                )
            if value["kind"] == "interface-test":
                return (
                    head
                    + f"\n{len(value['cases'])} cases; {len(value['witnesses'])} counterexamples; {value['repeated_argument_groups']} repeated-argument groups\n{value['scope']}"
                )
            if value["kind"] == "explanation":
                result = value["result"]
                effect = result.get("effect")
                tail = (
                    f"\nNative-primal matched: {effect['primal_logits_exact']}; gradient norm={effect['gradient_norm']:.6g}"
                    if effect
                    else f"\n{len(result.get('mlp', {}).get('neurons', []))} selected neurons; attention={result.get('attention', {}).get('probability_authority', 'unavailable')}"
                )
                details = []
                if value["view"] == "mlp":
                    details = [
                        f"  neuron {n['neuron']}: gate={n['gate']:.5g} up={n['up']:.5g} product={n['product']:.5g}"
                        for n in result["mlp"]["neurons"][:3]
                    ]
                elif value["view"] == "attention" and result["attention"].get("heads"):
                    details = [
                        f"  head {h['head']} KV{h['kv_group']}: "
                        + ", ".join(f"row {s['row']}={s['mass']:.3g}" for s in h["top_sources"][:3])
                        for h in result["attention"]["heads"][:3]
                    ]
                elif effect:
                    details = [
                        f"  neuron {n['neuron']}: product partial={n['product_partial']:+.5g}, activity projection={n['local_activity_projection']:+.5g}"
                        for n in effect["neurons"][:3]
                    ]
                elif "frequency" in result:
                    f = result["frequency"]
                    details = [
                        f"  {f['window']} source positions; return closure={f['return_closure_max_abs']:.3g}",
                        "  Source correlation view; one receiver, not semantic frequency bands.",
                    ]
                return (
                    head
                    + f"\nView: {value['view']}; layer={result['layer']}"
                    + tail
                    + ("\n" + "\n".join(details) if details else "")
                    + "\nUse --full or --json for operands, signed terms and parameter references."
                )
            return head
        if value.get("authority") == "original-model-view":
            return f"{value['tensor']} selector={value['selector']}\nshape={value['shape']} {value['dtype']}\nSample: {value['sample']} (transient)"
        if value.get("ok") is False or "error" in value:
            return "error: " + str(value.get("error", "operation failed"))
        if "runtime" in value and "state" in value:
            return str(value["state"])
        if "path" in value and "value" in value:
            return f"{value['path']}: {summary(value['value'], depth=depth + 1)}"
        if "shape" in value:
            fields = [
                f"{key}={value[key]}"
                for key in ("shape", "dtype", "norm", "min", "max")
                if key in value
            ]
            if "sha256" in value:
                fields.append("sha256=" + short_ref(value["sha256"]))
            return " ".join(fields)
        if "cuts" in value:
            cuts = value["cuts"]
            return f"{len(cuts)} cuts: " + ", ".join(
                short_ref(item.get("cut_id")) for item in cuts[-8:]
            )
        if "cut_id" in value:
            flags = [
                f"{key}={value[key]}"
                for key in ("name", "steps", "stopped", "committed", "aborted", "exact")
                if key in value
            ]
            return "cut=" + short_ref(value["cut_id"]) + (" " + " ".join(flags) if flags else "")
        if depth >= 2:
            return f"object ({len(value)} fields)"
        fields = list(value.items())[:6]
        text = "; ".join(f"{key}: {summary(item, depth=depth + 1)}" for key, item in fields)
        return text + (f"; ... ({len(value)} fields)" if len(value) > 6 else "")
    if isinstance(value, (list, tuple)):
        if depth >= 2:
            return f"[{len(value)} items]"
        parts = [summary(item, depth=depth + 1) for item in value[:6]]
        return (
            "[" + ", ".join(parts) + (f", ... ({len(value)} items)" if len(value) > 6 else "") + "]"
        )
    return json.dumps(value, default=str)


def cheat_sheet(counter=False):
    boundary, path, edit = (
        ("step", "value", "10") if counter else ("layer", "residual", "0.9 scale")
    )
    return (
        f"Motion: step | continue | pause\n"
        f"Breaks: break {boundary} 1 | watch {path}\n"
        "State:  checkpoint parent | fork candidate | restore parent\n"
        f"Edits:  set {path} {edit} | preview | commit | abort\n"
        "Views:  inspect memory | inspect --full | cuts\n"
        "Help:   help | history | reverse-step | quit"
    )
