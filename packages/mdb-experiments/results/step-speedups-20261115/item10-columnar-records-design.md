---
title: "Item 10 design note: columnar numbers in mdb records (not implemented on mdb-step-speedups)"
date: 2026-11-15
type: design-note
status: proposal
tags: [mdb, records, schema, design-note]
---

# Item 10: columnar numbers in records (design note, not implemented)

Status: not implemented on `mdb-step-speedups`. The task allowed item 10 only if it keeps record ids stable or bumps the schema cleanly with a migration. Neither fits in this branch without a decision from the owner, for the reasons below. Nothing here was measured on this branch except what is labelled MEASURED.

## The problem (from the review, CODE-READ)

`mdb.core.records.num(value, authority, proof=...)` (`core/records.py:225`) makes one dict per number: `{"value", "authority", "proof"?, "unit"?, "pointer"?}`. The review MEASURED 69 bytes per `num(...)` against 8 bytes of payload, so the 64 KiB record cap (`MAX_RECORD_BYTES`, checked on `stored_bytes`) holds about 900 numbers. A 28-layer x 32-head grid is 896 numbers, so one grid fills a record and goes to a pointer. Each record also embeds the full `execution` dict that the session row already holds.

## Why it cannot be done id-stable

A record id is the sha256 of the record's canonical JSON (`seal_record`, schema `mdb-record-v2`). Any change in how a number is written (a column instead of per-number dicts) changes the bytes, so every record that holds a number gets a new id. There is no encoding of a column that canonicalises to the same bytes as the per-number dicts. So item 10 is a schema change by construction: `mdb-record-v3`.

## Proposed v3 (for the owner)

1. A new numeric claim kind beside `num`, not instead of it:
   `{"kind": "nums", "authority": "derived-exact", "proof": "P15", "unit": null, "shape": [28, 32], "dtype": "float64", "values": [...]}`. One authority and proof per column. Inline up to a bound (proposal: 4096 values, about 40-80 KB of JSON at full float repr, so the bound has to be in bytes, not values; or switch inline floats to a fixed-width hex-of-float64 string for exactness and size).
2. Above the bound, `values` becomes a tensor pointer (`{"kind": "tensor", "ref", "sha256", "shape", "dtype"}`) to a safetensors blob in the project store, the same pointer type `each` results already use, instead of JSON text wrapped in a uint8 tensor (`core/bounded.py`).
3. Records bind to the session by `session_id` plus `sha256(canonical(execution))` instead of embedding `execution` (saves about 0.8 KB per record, MEASURED by the review on a tiny Qwen2).
4. `schema: "mdb-record-v3"`; readers accept v2 and v3; writers write v3 only behind a flag until the owner flips it.

## Migration

- v2 records stay valid and readable forever (ids are content hashes; they cannot be rewritten in place). `index.get_record` and `verify` keep the v2 path.
- A `mdb records migrate` op would write a v3 twin of a v2 record with a `supersedes: <v2 id>` pointer, so citations of the v2 id still resolve. No automatic rewrite.
- Tests: golden v2 fixtures must keep verifying; a v3 round-trip (num grid -> nums column -> same float64 values bitwise); a v3 record above the inline bound must store a blob and refuse a tampered blob.

## What it would change for users

Every forecast, branches, sweep and symbol-table record id changes under v3. Anything that cites record ids (posts, preregs, the release-gate logs) keeps pointing at v2 records, which stay valid. That is why this needs an owner decision and a version bump (0.3.0), and is not part of the speedups branch, whose hard gate is bit-identical record ids.
