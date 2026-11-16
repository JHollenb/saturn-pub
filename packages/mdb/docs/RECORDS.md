# Records: format, ids, bounds and migration

A record is a sealed JSON result: a forecast, a `whatif`, a symbol table, a branches or pool
record, the summary of an `each`, a trace. Its id is the sha256 of its stored form, so a record
cannot change after it is sealed without changing its id. This page describes schema
`mdb-record-v3`, written from mdb 0.3.0, and how records written by earlier versions
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

What a sink line, a project index row and the id cover is the stored form,
`records.stored_form(record)`: canonical JSON (sorted keys, no spaces, UTF-8) of the record
with two changes.

1. `execution` is replaced by `execution_sha256`, the sha256 of the execution profile's
   canonical JSON. The profile is the session's and is the same for every record of a session
   (about 0.8 KB on a tiny Qwen2), so it is stored once: in the project's session row, and as
   one `{"mdb_execution": "mdb-execution-v1", "sha256", "execution"}` line per sink file,
   written before the first record that needs it.
2. The `values` of an inline column are the base64 of their little-endian bytes: 8 bytes per
   float64 value, 10.7 characters of base64, against about 70 bytes for one `num` per value.

`id` and `digest` are the sha256 of that stored form without `id` and `digest`. The id still
covers every field, the execution profile included, through its sha256: change any byte of the
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
execution line) or a directory searched for record files; other JSONL files are skipped.
For `NAME.jsonl`, `migrate` writes `NAME.v3.jsonl` beside it (or under `--out`, keeping the
directory layout) and `NAME.v3.map.json`:

```json
{"schema": "mdb-record-migration-v1", "source": "...", "source_sha256": "...",
 "target": "...", "target_sha256": "...", "records": 380, "migrated": 380,
 "already_v3": 0, "payloads": 2, "verified": 380, "map": {"<v2 id>": "<v3 id>", ...}}
```

Each v2 record is verified, re-sealed as v3 with the same content and no custody added, and the
new file is then read back: every record must equal its source in every field except `schema`,
`id` and `digest` (canonical JSON, so every number is bit-identical), or nothing is written.
Payload lines are copied byte for byte. The source file is never changed, and an existing
target is refused unless `--force`.

The same in Python: `mdb.core.records.migrate_record(record)` and
`mdb.core.migrate.migrate_file(path)`.

What migration does not do:

- It does not rewrite record ids that appear inside a body (a `watch-hits` record's `trace`, a
  value's `origin.record`). They are content and keep the v2 id; the map file resolves them,
  and the v2 file stays readable.
- It does not rewrite project indexes or core files. Their v2 records are read as they are.
- It does not turn lists of `num`s into columns. That would change the content.

## What changed for ids

Every record sealed by mdb 0.3.0 has a new id, even with identical content, because the stored
form changed. Example: with torch 2.13 and transformers 5.13.1, `examples/00-hello-tiny` seals
its forecast as `f30d4d21ce1341f0` under mdb 0.2.1 and as `d0c1d6c733c4bfa1` under 0.3.0;
everything it prints besides that id is unchanged, and `mdb records migrate` maps the first to
the second. (The forecast record names the model identity, so other library versions give other
ids; `expected_output.txt` was made with other versions.)
