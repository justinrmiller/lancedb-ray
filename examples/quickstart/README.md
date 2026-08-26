# Quickstart

The smallest end-to-end demonstration of what `lancedb-ray` guarantees. It
generates a synthetic vector dataset, writes it through Ray, and then asserts
the properties that make the local write path worth using at all.

No external services, no credentials, no model downloads — it runs against a
temporary local LanceDB directory.

## Run it

```bash
python examples/quickstart/quickstart.py
```

Useful flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--rows` | `100000` | Rows to generate |
| `--dim` | `128` | Embedding dimensionality |
| `--blocks` | `8` | Ray blocks, i.e. how wide the write fans out |
| `--uri` | temp dir | Write to a specific LanceDB directory instead of a temp one |

## What it proves

The script asserts three things rather than just printing them, so it fails
loudly if a change breaks them.

**The write fans out.** Eight Ray blocks produce more than one Lance fragment,
which is what tells you the work actually ran in parallel across workers rather
than funnelling through the driver.

**The write is atomic.** Those fragments all land in a *single* new table
version. This is the property that matters most in a distributed write: a
reader either sees none of the data or all of it, never a half-finished table.
Workers write fragments independently and the driver commits them in one
transaction at the end.

**Reads come back fragment-parallel.** Reading the table yields multiple blocks,
tracking the fragments on disk, so downstream Ray stages get real parallelism
instead of one giant block.

It then does a projected, filtered read to show that `columns` and `filter` are
pushed down into the scan rather than applied after loading everything.

## Expected output

```
Writing 100,000 rows across 8 blocks to /tmp/lancedb_ray_demo_xxxx
  rows written : 100,000
  fragments    : 8  (parallel write)
  versions     : 1  (atomic commit)

Reading back...
  rows read    : 100,000
  blocks       : 8  (fragment-parallel read)
  filtered rows: 1,000 with columns ['id', 'label']

All checks passed.
```

The exact fragment and block counts depend on how many CPUs Ray sees, but
`versions` must always be `1`.
