# Recomputing birefringence images (`ppm-recompute-biref`)

A setup + operator guide for regenerating PPM birefringence images that were written
with the **old luminance-first** algorithm, using the current **per-channel color-RMS**
method.

- **Script:** `ppm_library/tools/recompute_biref.py`
- **Command (installed):** `ppm-recompute-biref`
- **Module form (no install script on PATH):** `python -m ppm_library.tools.recompute_biref`

---

## 1. What it does and why

Older birefringence (`*_biref*`) images were computed by converting each polarization
angle to luminance first and then taking the normalized difference. Structures whose
*color* rotated between the two angles while their weighted luminance stayed nearly equal
would cancel to near-zero and show up as broken / discontinuous segments.

The current method computes the normalized difference **per RGB channel** and combines
them as a root-mean-square magnitude:

```
biref = sqrt(mean(nd_R^2, nd_G^2, nd_B^2))
```

This keeps those structures continuous. This tool re-derives each old biref from its two
original angle source tiles using the *current* library function
(`TifWriterUtils.ppm_normalized_difference_abs`), so the output is byte-for-byte
consistent with what fresh acquisitions now produce.

> **Hard requirement:** recomputation is only possible while **both** angle source tiles
> still exist next to the biref. Those originals are large and are often deleted after
> stitching. Where they are gone, the biref **cannot** be regenerated -- the tool reports
> those files as un-recomputable and moves on. There is no way around this; the source
> pixels are the only place the information lives.

---

## 2. The file-naming convention (how "paired images" are matched)

The tool walks a folder **recursively**, finds every file whose name contains `biref`,
and for each one looks in the **same directory** for its positive- and negative-angle
source tiles. Pairing is purely by filename:

| Role | Filename pattern | Example (`+7`/`-7` degrees) |
|---|---|---|
| Birefringence (the file being replaced) | `<base>_<angle>_biref<tail>.ome.tif` | `slideA_7_biref_001.ome.tif` |
| I(+) positive-angle source | `<base>_<angle><tail>.ome.tif` | `slideA_7_001.ome.tif` |
| I(-) negative-angle source | `<base>_-<angle><tail>.ome.tif` | `slideA_-7_001.ome.tif` |

- The **angle** is whatever number sits immediately before `_biref` (integer or decimal,
  e.g. `5`, `7.5`).
- The **tail** is everything after the angle (e.g. `_001`) and must match across the trio.
- Supported extensions: `.ome.tif`, `.ome.tiff`, `.tif`, `.tiff`.

So "point it at a folder of paired images" means: a folder (or tree of folders) in which
each biref still sits alongside its two source angle tiles under this naming scheme. You
do not have to group them yourself -- the tool pairs them automatically. Nested
subfolders are fine; each biref is paired within its own directory.

If a biref's name does not match the pattern, or its sources are missing, that file is
skipped/reported and the rest of the batch continues.

---

## 3. Installing on a new (high-RAM) computer

These images can be tens to hundreds of gigapixels, so run this on a machine with plenty
of RAM (see the RAM section below for the actual math). Setup is a normal Python install.

### 3.1 Prerequisites

- **Python 3.10 or newer** (`python3 --version`).
- Enough free RAM for at least one wide row-strip of two source tiles plus the output
  (Section 5).
- Fast local disk with room for the recomputed outputs (roughly the size of the biref
  files you are regenerating).

### 3.2 Get the code and install the package

```bash
# 1. Get ppm_library onto the machine (clone the repo, or copy the ppm_library folder)
git clone <ppm_library-repo-url> ppm_library
cd ppm_library

# 2. (Recommended) create an isolated environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install the package (pulls numpy, tifffile, imagecodecs, opencv, etc.)
pip install .
#    ...or for an editable/dev checkout:  pip install -e ".[dev]"
```

### 3.3 (`zarr` is included automatically)

The streaming reader opens each source tile through `tifffile`'s zarr interface, so
`zarr` is required at runtime. It is declared in `pyproject.toml`, so `pip install .`
pulls it in -- no separate step needed. If you are running against an **older checkout**
whose `pyproject.toml` predates that fix and you hit
`ModuleNotFoundError: No module named 'zarr'`, install it manually:

```bash
pip install zarr
```

### 3.4 Verify the install

```bash
ppm-recompute-biref --help
# If the console script isn't on PATH (no `pip install`, or PATH not refreshed):
python -m ppm_library.tools.recompute_biref --help
```

---

## 4. Running it

**Always dry-run first** to see what will be paired, recomputed, or skipped -- nothing is
written in a dry run:

```bash
ppm-recompute-biref "/path/to/images" --dry-run
```

Read the summary line at the end (`N recomputed, M un-recomputable, ...`). If the pairing
and counts look right, run for real.

### 4.1 Safe default -- write new files alongside the originals

```bash
ppm-recompute-biref "/path/to/images"
```

Each recomputed biref is written next to the original with a `_colorrms` suffix
(`slideA_7_biref_001_colorrms.ome.tif`), leaving the old file untouched. This is the
recommended first real run: you can compare old vs new before committing.

### 4.2 Overwrite the originals in place

Once you trust the results, regenerate over the originals:

```bash
ppm-recompute-biref "/path/to/images" --overwrite
```

### 4.3 Re-running / resuming

By default an existing output is left alone (`[exists]`), so re-running the same command
resumes where it stopped and skips work already done. Force a redo with `--force`.

### 4.4 All options

| Flag | Default | Purpose |
|---|---|---|
| `folder` (positional) | -- | Folder to scan **recursively** for biref images. |
| `--suffix` | `_colorrms` | Suffix for the recomputed output file. |
| `--overwrite` | off | Overwrite the original biref in place (ignores `--suffix`). |
| `--dry-run` | off | Report what would happen; write nothing. |
| `--force` | off | Recompute even if the output already exists. |
| `--min-intensity` | `0.0` | Dark-region mask threshold (8-bit scale); pixels dimmer than this are masked out of the biref. |
| `--strip` | `2048` | Row-strip height for streaming -- **lower this to cut RAM** on very wide images. |
| `--mem-budget-gb` | `3.0` | Max RAM for the *output* array before switching to a memory-mapped write. |

Exit code is `0` on success, `1` if any file failed, `2` if the folder path is invalid.

---

## 5. RAM and performance (why a high-RAM machine)

Two things consume memory per file:

1. **Row strips of the two source tiles.** Each strip is read as float32 RGB for *both*
   the positive and negative tile:

   ```
   strip RAM  ~=  strip_rows * image_width * 3 channels * 4 bytes * 2 tiles
   ```

   For a 100,000-px-wide slide at the default `--strip 2048`, that is roughly
   `2048 * 100000 * 3 * 4 * 2 ~= 4.9 GB` held at once -- independent of how the output is
   written. **Wide images are the RAM driver.** If you hit memory pressure, reduce
   `--strip` (e.g. `--strip 512`); it trades a bit of speed for a ~4x smaller strip
   footprint.

2. **The output array.** If the full output (`height * width * 2 bytes`, uint16) fits
   within `--mem-budget-gb`, it is built in RAM and written once. If it exceeds the
   budget, the tool automatically memory-maps the output to disk and fills it strip by
   strip -- so arbitrarily large slides still work, just with more disk I/O. Raise
   `--mem-budget-gb` on a big-RAM box to keep more outputs fully in memory (faster).

**Rule of thumb:** for very large/wide slides, prefer a machine with 32-64 GB+ RAM, keep
the default strip (or lower it if you go wider than ~100k px), and give
`--mem-budget-gb` a few GB less than your free RAM so the strip buffers have headroom.

---

## 6. Reading the output

Per-file status lines:

- `[dry   ]` -- would recompute (dry run only).
- `[ok    ]` -- recomputed and written (shows `width x height`).
- `[exists]` -- output already present; skipped (use `--force` to redo).
- `[skip  ]` -- filename did not match the biref pattern.
- `[MISSING]` -- one or both source angle tiles are gone; **cannot** recompute.
- `[FAIL  ]` -- an error occurred on this file; the batch continues.

Final summary:

```
summary: 42 recomputed, 5 un-recomputable (sources gone), 3 skipped, 0 failed
```

`un-recomputable` is expected wherever the source tiles were deleted after stitching --
those birefs are permanently frozen with the old algorithm unless the originals are
restored from backup.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'zarr'` | Only on an older checkout predating the dependency fix -- `pip install zarr` (Section 3.3). |
| `ppm-recompute-biref: command not found` | Console script not on PATH; use `python -m ppm_library.tools.recompute_biref ...`, or re-activate the venv. |
| Everything reports `[MISSING]` | The source angle tiles are not next to the biref, or their names do not follow `<base>_<angle><tail>` / `<base>_-<angle><tail>`. Check one trio by hand against Section 2. |
| Everything reports `[skip  ]` | Biref filenames don't contain `biref` immediately after `_<angle>_`, or use an unsupported extension. |
| Process killed / MemoryError on wide slides | Lower `--strip` (e.g. `512`) and/or lower `--mem-budget-gb`; run on a higher-RAM machine (Section 5). |
| Compression / codec errors reading a tile | Ensure `imagecodecs` installed (it is a declared dependency; `pip install imagecodecs` if missing). |

---

## 8. Dependency note

`ppm_library/tools/recompute_biref.py` imports `zarr` at runtime (in `_read_strip`).
`zarr>=2.16` is declared under `[project.dependencies]` in `pyproject.toml`, so
`pip install .` is sufficient on a new computer. (Prior to that fix, `zarr` had to be
installed manually.)
