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
    return f"({runtime.get('branch', 'main')}) " + " ".join(fields)


def summary(value, *, depth=0):
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    if isinstance(value, dict):
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
