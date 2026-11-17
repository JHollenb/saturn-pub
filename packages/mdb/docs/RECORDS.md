# Records: format, ids, bounds and migration

A record is a sealed JSON result: a forecast, a `whatif`, a symbol table, a branches or pool
record, the summary of an `each`, a trace. Its id is the sha256 of its stored form, so a record
cannot change after it is sealed without changing its id. This page describes schema
`mdb-record-v3`, written from mdb 0.2.2, and how records written by earlier versions
(`mdb-record-v2`) are read and migrated.

## The record you work with

`seal_record`, `sink.records()`, `index.get_record` and every verb that seals a record give you
the same mapping in v3 as in v2:

```python
{"schema": "mdb-record-v3", "id": ..., "digest": ...,      # digest == id
 "session_id": ..., "kind": "forecast", "model_identity": {...}, "execution": {...},
 "authority": "derived-exact", "pointers": [...], "body": {...}}
```

Inside `body`, every number is labelled (see [Authority labels](../README.md#authority-labels)):
one number is `num(value, authority, proof=, unit=, pointer=)`, many numbers with one label are a
column, `nums(values, authority, proof=, unit=, shape=, dtype=)`:

```python
from mdb.core.records import nums, nums_values

grid = nums(effects, "derived-exact", proof="P15")    # a (28, 32) array, list or tensor
# {"kind": "nums", "authority": "derived-exact", "proof": "P15",
#  "shape": [28, 32], "dtype": "float64", "values": [... 896 floats, row-major ...]}
nums_values(record["body"]["grid"])                    # numpy array, shape (28, 32)
```

`dtype` is `float64` (the default for floats), `float32` (every value must be exactly a float32;
the default for float32, float16 and bfloat16 tensors) or `int64` (the default for integers).
Values are finite. A column holds up to 65,536 values inline. Pass `store=` (a project or its
store) for more, or `inline=False` for fewer: the values then go to a tensor blob in the
project store, `values` is a tensor pointer, and `nums_values(claim, store=...)` reads the blob
back and checks its sha256.

## The stored form

The id covers the stored form's parsed value, not the line's bytes: the form is
`records.stored_form(record)`: canonical JSON (sorted keys, no spaces, UTF-8) of the record
with two changes.

1. `execution` is replaced by `execution_sha256`, the sha256 of the execution profile's
   canonical JSON. The profile is the session's and is the same for every record of a session
   (about 0.8 KB on a tiny Qwen2), so it is stored once: in the project's session row, and as
   one `{"mdb_execution": "mdb-execution-v1", "sha256", "execution"}` line per sink file,
   written before the first record that needs it.
2. The `values` of an inline column are the base64 of their little-endian bytes: 8 bytes per
   float64 value, 10.7 characters of base64, against about 70 bytes for one `num` per value.

`id` and `digest` are the sha256 of that stored form without `id` and `digest`. Key order,
whitespace and `\u` escapes of a stored line are not covered (mdb writes canonical text but does
not require it when reading); a line with a duplicate key is refused on every read path, and a
line longer than the 1 MiB bound plus 4 KiB is refused before it is parsed. The id still
covers every field, the execution profile included, through its sha256: change any value of the
content or the profile and the id changes.

Reading a stored record back needs its execution profile:

```python
from mdb.core.records import load_record

record = load_record(line, execution=profile)          # or executions={sha256: profile}
sink.records()                                          # does this for you
```

`load_record` refuses truncated JSON, base64 that is not canonical or not the column's length,
non-finite values, a profile whose sha256 differs, and any record whose content no longer
matches its id.

## Bounds

| | v3 | v2 |
|---|---|---|
| stored form | 1 MiB | 64 KiB |
| JSON values outside columns | 65,536 | 4,096 |
| items in one inline list (nested lists share it) | 64 | 64 |
| text | 4 KiB per string | 4 KiB per string |
| values in one inline column | 65,536 | (no columns) |

The 64-item list bound is about readable structure: a table of numbers is a column, not a list
of `num`s. Anything larger goes by pointer (a tensor blob or a `bounded.put_json` payload) as
before.

## Checkpoint records (`checkpoint-state`)

A checkpoint (`run.dump`, `dbg.checkpoint`, `Run.checkpoint(durable=...)`; see
[Checkpoints](BRANCHES-POOLS-HOLDS.md#checkpoints-enginecheckpointpy-core-files)) is sealed as a
v3 record of kind `checkpoint-state`. The tensors are not in the record. It points at them:

```python
{"schema": "mdb-checkpoint-v2", "name": ..., "where": "t4.L1.p", "port": "p",
 "cut": {"fingerprint", "transitions", "row", "label"},     # the cursor's boundary cut
 "root": {"fingerprint", "transitions", "row", "label"},    # the first kept mark
 "history": "row", "marks": 3, "stored_cuts": 3,
 "prompt": {"tokens": [...64], "length", "sha256"}, "procedure": {...}, "shape": {...},
 "source": {"dtype", "device"},
 "lineage": {"root": "prompt" | "checkpoint", "checkpoints": [ids], ...},
 "recipe": {"sha256", "path_edits", "staged"}, "edited": bool,
 "holds": [{"name", "address", "dropped"}], "captured": [addresses],
 "document": {"state": "stored" | "file", "sha256", "bytes", "ref"?, "tensor"?},
 "tensors": {"count", "sha256": ["<60 sha256 hex per string>", ...] | None, "parts_sha256"},
 "bytes": num(..., "measured"), "custody": {...}}
```

`document` is the state document (cuts, marks, edits, the recipe), sealed by its sha256: a JSON
blob in the project store, or a member of the checkpoint file. `tensors.sha256` lists the sha256
of every tensor part when there are at most 3,600, packed 60 per string because a record list
holds at most 64 items. `parts_sha256` is the sha256 of the sorted list either way. Loading
refuses a record whose parts, document, prompt, fingerprints or recipe differ from what it
seals. `lineage.root` is `prompt` when the cut's fingerprints chain back to a run of the prompt.
It is `checkpoint` for a replica or a legacy core file, which root a new lineage. The older
`core-file` kind (legacy v1 core files) is still declared and read.

The seal gives integrity, not authenticity: it binds the document (edit history and lineage
included) to the record id, but anyone can seal a document whose history was stripped. So a
record sealed about a run rehydrated from a checkpoint names it until the path is verified by
recompute: `procedure.from_checkpoint` (records that carry a procedure) and
`custody.from_checkpoint` (every record sealed about that run: the sealer passes
`seal_record(run=...)`; a record about several such runs lists them under `checkpoints`) are
`{"checkpoint": "<record id>", "path": "unverified" | "verified"}` (a replica adds `replica` and
`null_check`).

## Records written by mdb 0.2.1 and earlier (v2)

v2 records stay valid and are read everywhere v3 records are: `verify_record`, `sink.records()`
(a file may hold both), the project index, core files and `mdb serve`. Their bounds and their
ids are unchanged; nothing rewrites them in place. In v2 the stored form is the canonical JSON
of the record as shown above, `execution` included, and the id is its sha256. Legacy
`mdb-circuit-investigation-v1` records are read as before.

## Migrating files to v3

```
mdb records migrate PATH [PATH ...] [--out DIR] [--force]
mdb records verify PATH [PATH ...]
```

`PATH` is a record file (a sink, or any JSONL file whose every line is a record, payload or
execution line) or a directory searched for record files (a `*.jsonl` that mentions a record schema or an
execution line; other JSONL files are skipped). A record file that is damaged, altered (an
unknown `schema`, a changed body) or truncated is never skipped: it is named on stderr as
refused and the command exits 1. `verify` also reads `.v3.jsonl` files.
For `NAME.jsonl`, `migrate` writes `NAME.v3.jsonl` beside it (or under `--out`, keeping the
directory layout) and `NAME.v3.map.json`:

```json
{"schema": "mdb-record-migration-v1", "source": "...", "source_sha256": "...",
 "target": "...", "target_sha256": "...", "records": 380, "migrated": 380,
 "already_v3": 0, "payloads": 2, "verified": 380, "check": "every record read back ...",
 "map": {"<v2 id>": "<v3 id>", ...}}
```

Each v2 record is verified, re-sealed as v3 with the same content and no custody added, and the
new file is then read back: every record must equal its source in every field except `schema`,
`id` and `digest` (canonical JSON, so every number is bit-identical), or nothing is written.
Payload lines are copied byte for byte. The source file is never changed (a target that is the
source, or does not end in `.v3.jsonl`, is refused even with `--force`), and an existing target is refused unless `--force`. The map is
written (atomically) before the target.

The same in Python: `mdb.core.records.migrate_record(record)` and
`mdb.core.migrate.migrate_file(path)`.

What migration does not do:

- It does not rewrite record ids that appear inside a body (a `watch-hits` record's `trace`, a
  value's `origin.record`). They are content and keep the v2 id; the map file resolves them,
  and the v2 file stays readable.
- It does not rewrite project indexes or core files. Their v2 records are read as they are.
- It does not turn lists of `num`s into columns. That would change the content. A v2 body that
  already holds a column-shaped dict (`{kind: "nums", shape, dtype, values}`) is read as a v3
  column after migration; one with `dtype: "float64"` and integer values is refused.

## What changed for ids

Every record sealed by mdb 0.2.2 has a new id, even with identical content, because the stored
form changed. Example: with torch 2.13 and transformers 5.13.1, `examples/00-hello-tiny` seals
its forecast as `f30d4d21ce1341f0` under mdb 0.2.1 and as `d0c1d6c733c4bfa1` under 0.2.2;
everything it prints besides that id is unchanged, and `mdb records migrate` maps the first to
the second. (The forecast record names the model identity, so other library versions give other
ids; `expected_output.txt` was made with other versions.)
