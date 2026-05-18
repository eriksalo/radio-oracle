# Jetson HNSW load failure — wikipedia + gutenberg

## Symptom

After rsyncing `data/chroma/` from the workstation to the Jetson (`/opt/radio-oracle/data/chroma/`), opening the `wikipedia` or `gutenberg` collection raises:

```
chromadb.errors.InternalError: Error executing plan: Error sending backfill request to compactor:
  Error constructing hnsw segment reader: Error creating hnsw segment reader: Error loading hnsw index
```

The other four collections (`wikibooks`, `wikimed`, `ifixit`, `crashcourse`) open fine on the Jetson. All six open fine on the workstation. md5sums of every HNSW file match byte-for-byte between the two machines.

## Earlier (incorrect) hypothesis

A prior Claude session on the Jetson speculated this was an x86 → aarch64 portability issue — different endianness, struct padding, or pickle layout. That guess is wrong:

- x86_64 and aarch64 are both little-endian; hnswlib uses fixed-size POD types with `writeBinaryPOD`.
- File md5s match between hosts, so transfer integrity is fine.
- Same chromadb version (1.5.x) on both sides — no on-disk format break.

## Actual cause

chroma-core's hnswlib fork allocates the level-0 element buffer during `loadPersistedIndex` as:

```cpp
data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
```

`max_elements_` is the **preallocated capacity** written into `header.bin`, not the actual element count.

For our two big collections:

| Collection | cur_element_count | max_elements | size_data_per_element | malloc(max × sde) |
|------------|-------------------|--------------|-----------------------|--------------------|
| wikipedia  | 11,476,005        | 16,777,216   | 1,676                 | **28.1 GB**        |
| gutenberg  | 10,301,735        | 16,777,216   | 1,676                 | **28.1 GB**        |
| wikibooks  |    313,401        |    524,288   | 1,676                 |    878 MB          |
| wikimed    |    258,730        |    262,144   | 1,676                 |    439 MB          |
| ifixit     |    181,502        |    262,144   | 1,676                 |    439 MB          |
| crashcourse|      1,654        |      2,048   | 1,676                 |    3.4 MB          |

The Jetson Orin Nano Super has 7.4 GB RAM + ~12 GB swap (8 GB swapfile + 6× 634 MB zram). With `vm.overcommit_memory=0` (heuristic mode, the default), a 28 GB allocation is refused.

Verified directly:

```bash
$ python3 -c '
import ctypes
libc = ctypes.CDLL("libc.so.6"); libc.malloc.restype = ctypes.c_void_p
print(libc.malloc(16777216 * 1676))  # 28.1 GB → None  (FAILS)
print(libc.malloc(11476005 * 1676))  # 19.2 GB → addr  (succeeds)
'
None
281471435911184
```

`malloc` returning `None` → hnswlib's `data_level0_memory_` is NULL → load throws → chromadb wraps it as `Error loading hnsw index`.

## Fix options

### 1. Patch header.bin (recommended first step)

Set `max_elements_ = cur_element_count` in `header.bin` for wikipedia + gutenberg. Drops `malloc` to the actual on-disk size:

- wikipedia: 19.2 GB
- gutenberg: 17.3 GB

Both fit a `malloc` call on the Jetson (verified). Side effect: those collections become read-only — any further insert past `cur_element_count` would require a rebuild. Acceptable since ingest happens exclusively on the workstation.

Field offset in `header.bin` (chroma fork layout, 100 bytes total):

```
0x00  u32 LE  PERSISTENCE_VERSION       (1)
0x04  u64 LE  offsetLevel0_             (0)
0x0C  u64 LE  max_elements_             <-- patch this to cur_element_count
0x14  u64 LE  cur_element_count         (read this first)
0x1C  u64 LE  size_data_per_element_    (1676)
0x24  u64 LE  label_offset_             (1668)
0x2C  u64 LE  offsetData_               (132)
0x34  i32 LE  maxlevel_
0x38  u32 LE  enterpoint_node_
0x3C  u64 LE  maxM_                     (16)
0x44  u64 LE  maxM0_                    (32)
0x4C  u64 LE  M_                        (16)
0x54  f64 LE  mult_                     (~0.36)
0x5C  u64 LE  ef_construction_          (100)
```

### 2. Add swap to ~64 GB on the Jetson

`malloc` succeeding only reserves address space — actual page commits happen during the `ifstream.read` that follows. After both collections are loaded, the resident set is ~36.5 GB. The Jetson cannot hold both in 7.4 GB RAM + 12 GB swap simultaneously. Adding `/swapfile2` (54 GB) on the 393 GB-free root filesystem gives headroom.

### 3. Architectural: switch to a mmap-capable index (longer term)

`malloc + read` is the wrong shape for a 20 GB index on an 8 GB device. Each query touches a small random subset of `data_level0.bin` (a few thousand 1.6 KB elements per `ef_search=100` traversal) — naturally suited to mmap. Even with option 2 applied, queries will be slow (~5–30 s) under swap thrashing.

Candidates:

- **FAISS IVF-PQ** — quantized vectors, ~64 B/element instead of 1.6 KB. wikipedia drops from 19 GB to ~700 MB; fits in RAM.
- **USearch** with `--storage view` / mmap mode.
- A small chromadb fork that mmaps `data_level0.bin` instead of mallocing.

## Status

**Diagnose-only, 2026-05-17.** No changes applied to the Jetson yet. Decision on which fix to apply pending.
