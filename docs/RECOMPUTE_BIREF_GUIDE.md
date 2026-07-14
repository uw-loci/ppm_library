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
original angle source images using the *current* library function
(`TifWriterUtils.ppm_normalized_difference_abs`), so the pixel values are byte-for-byte
consistent with what fresh acquisitions now produce.

**Output format matches the production stitched biref.** The recomputed file is a
**pyramidal OME-TIFF** with the same structure the QPSC stitcher writes: single-channel
16-bit (uint16, minisblack, big-endian), 512-pixel tiles, LZW compression, factor-2
pyramid levels down to the tile size, BigTIFF when the image is large, and the physical
pixel size carried over from the source. So a regenerated slide drops into QuPath and
loads at full pyramid speed exactly like the original. Tile size and compression are
adjustable (`--tile`, `--compression`) if you need to match a non-default setup.

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

### 3.5 Trying it on the acquisition (PPM) machine

On the PPM microscope you do **not** set up from scratch: `ppm_library` is already
installed there (it computes birefringence during acquisition). You just **update** it. In
the environment where `ppm_library` lives:

```bash
cd <path-to-ppm_library-on-the-scope>
git pull
pip install -e .          # or: pip install .
ppm-recompute-biref --help
```

That registers the `ppm-recompute-biref` command **and** installs the one new dependency,
`zarr`; `numpy`, `tifffile`, and `imagecodecs` (LZW) are already `ppm_library`
dependencies. A bare `git pull` is not enough -- the console command is a new entry point
and `zarr` is a new dependency, both added at install time. If you would rather not
reinstall the package:

```bash
pip install zarr
python -m ppm_library.tools.recompute_biref --help
```

Small test images need no RAM flags (the default in-RAM path handles them). Note that
recently acquired images were already written with the current per-channel method, so
recomputing them reproduces nearly identical values -- useful to confirm the tool runs and
emits a valid pyramidal OME-TIFF, but you will only see a real before/after difference on
older luminance-first images.

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
| `--strip` | `2048` | Row-strip height for streaming reads/downsampling -- **lower this to cut RAM** on very wide images. |
| `--tile` | `512` | Output tile size (matches the QPSC stitched biref). |
| `--compression` | `lzw` | Output compression: `lzw` (default, matches QPSC), `zlib`, `jpeg2000`, or `none`. |
| `--mem-budget-gb` | `3.0` | Hold the full-res biref in RAM up to this size; above it, use a temporary memory-mapped file beside the output. **Raise this on a high-RAM machine** to keep large slides fully in RAM (faster). |

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

2. **The full-resolution biref.** The single-channel uint16 biref is `height * width * 2
   bytes` (e.g. a 100,000 x 100,000 slide is ~20 GB; 200,000 x 200,000 is ~80 GB). If it
   fits within `--mem-budget-gb` it is built in RAM; the pyramid levels are then generated
   by strip-wise 2x2 averaging (each level is 1/4 the previous, so they add little). If it
   exceeds the budget, the tool builds the full-res biref in a **temporary memory-mapped
   file beside the output** and streams the pyramid write from there -- so arbitrarily
   large slides still work, at the cost of temp disk equal to the biref size (the temp is
   deleted when the file finishes). Raise `--mem-budget-gb` on a big-RAM box to keep the
   biref fully in RAM (faster, no temp file).

**On a high-RAM machine (e.g. 512 GB):** set `--mem-budget-gb` high (for example
`--mem-budget-gb 300`) so even very large slides stay in RAM. Keep the default `--strip`
(or lower it if a slide is wider than ~100k px). The strip buffers and the in-RAM biref
are the two consumers; with hundreds of GB you have ample headroom for both.

```bash
ppm-recompute-biref "/path/to/images" --mem-budget-gb 300
```

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
