"""Release-gate 0.2.0 custody probe: record-sink naming inside a real mrun job.

Checks (prints one CUSTODY_JSON line):
  * run_tag() carries MRUN_JOB_ID; records_to(dir) / records_to("x-{run}.jsonl") use it;
  * two "shard" processes in one job writing to one directory get distinct files, no lost records;
  * an existing non-empty file this process did not open is refused; append=True accepts it;
  * MDB_RECORDS=<dir> env sink gets a per-process file;
  * opening the same path twice in one process (does a record get written twice?).
No model is loaded.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("MRUN_ARTIFACTS_DIR") or os.environ.get("MDB_OUT") or "out") / "custody-probe"
OUT.mkdir(parents=True, exist_ok=True)


def seal(tag):
    from mdb.core import records
    try:
        records.declare_kind("rg-probe")
    except Exception:
        pass
    return records.seal_record(session_id="release-gate", kind="rg-probe", model_identity={"name": "none"},
                               execution={"device": "cpu"}, authority="measured",
                               body={"tag": tag, "t": str(time.time())})


def shard(i):
    from mdb.core import custody
    s = custody.records_to(str(OUT / "shared-dir") + "/")
    for k in range(5):
        seal(f"shard{i}-{k}")
    print(json.dumps({"shard": i, "path": str(s.path), "count": s.count}))


def env_shard():
    from mdb.core import custody
    seal("env")
    print(json.dumps({"env_sinks": [str(s.path) for s in custody.sinks()]}))


def main():
    from mdb.core import custody
    res = {"MRUN_JOB_ID": os.environ.get("MRUN_JOB_ID"), "run_tag": custody.run_tag()}
    res["tag_has_job_id"] = bool(res["MRUN_JOB_ID"]) and res["run_tag"].startswith(res["MRUN_JOB_ID"])
    s_dir = custody.records_to(str(OUT / "own-dir") + "/")
    s_run = custody.records_to(str(OUT / "named-{run}.jsonl"))
    seal("main-0")
    res["dir_sink"] = str(s_dir.path)
    res["run_sink"] = str(s_run.path)
    res["dir_sink_count"] = len(s_dir.records())
    s_dir.close(); s_run.close()
    # RG: a Path to a not-yet-existing directory (a Path cannot end in a separator)
    s_p = custody.records_to(OUT / "fresh-dir-as-path")
    res["path_to_new_dir_becomes"] = "file" if not str(s_p.path).endswith(".jsonl") else "dir"
    res["path_to_new_dir_sink"] = str(s_p.path)
    s_p.close()

    procs = [subprocess.run([sys.executable, __file__, "shard", str(i)], capture_output=True, text=True)
             for i in range(2)]
    shards = [json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 else {"err": p.stderr[-800:]}
              for p in procs]
    res["shards"] = shards
    paths = [s.get("path") for s in shards]
    res["shard_paths_distinct"] = len(set(paths)) == 2 and None not in paths
    files = sorted((OUT / "shared-dir").glob("*.jsonl"))
    res["shared_dir_files"] = [f.name for f in files]
    res["shared_dir_records"] = sum(len(custody.Sink(f).records()) for f in files)

    foreign = OUT / "foreign.jsonl"
    foreign.write_text('{"x": 1}\n')
    try:
        custody.records_to(foreign)
        res["foreign_refused"] = False
    except FileExistsError as e:
        res["foreign_refused"] = True
        res["foreign_msg"] = str(e)[:200]
    s_app = custody.records_to(foreign, append=True)
    seal("append")
    res["append_lines"] = len(foreign.read_text().splitlines())
    s_app.close()

    twice = OUT / "twice.jsonl"
    a = custody.records_to(twice)
    b = custody.records_to(twice)
    seal("twice")
    res["same_path_twice_lines"] = len(twice.read_text().splitlines())
    a.close(); b.close()

    env = {**os.environ, "MDB_RECORDS": str(OUT / "env-dir") + "/"}
    p = subprocess.run([sys.executable, __file__, "env"], capture_output=True, text=True, env=env)
    res["env_sink"] = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else p.stderr[-800:]
    print("CUSTODY_JSON " + json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shard":
        shard(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "env":
        env_shard()
    else:
        main()
