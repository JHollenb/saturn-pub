---
name: Bug report
about: Something in saturn-pub or mdb does not work as documented
title: "[bug] "
labels: bug
---

**Package and version**

- Package: `mdb-runtime` / `saturn-pub` / both
- Version or ref (`mdb version`, `python -c "import saturn_pub; print(saturn_pub.__version__)"`, or the tag / commit you installed):
- Python, OS, torch and transformers versions:
- Device and dtype (cpu / cuda / mps; bf16 / fp32):

**What you did**

The smallest reproduction you can make. A random tiny model (`mdb engine tiny:qwen2`, `mdb.open_tiny(...)`, `saturn_pub.adapters.decoder` tiny configs) is best, because anyone can run it without weights or a GPU.

```
commands or code
```

**What happened**

Full error text or the wrong output (record ids, numbers, the refusal message).

**What you expected**

And, if you can, which doc or docstring says so.

**Config**

Do you have an mdb config file or `MDB_*` variables set? The output of `mdb config show` helps (remove any paths you do not want to share).
