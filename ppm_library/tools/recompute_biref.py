#!/usr/bin/env python3
"""Recompute PPM birefringence images in a folder with the current per-channel method.

Walks a folder recursively, finds birefringence images (filename contains "biref"),
locates each one's positive/negative angle source tiles, and recomputes the biref
using the current ``ppm_library.imaging.writer.TifWriterUtils.ppm_normalized_difference_abs``
(per-channel color-RMS). Importing the library function guarantees the output matches
what fresh acquisitions now produce.

Source-file convention (matches the acquisition writer):

    <base>_<angle>_biref<tail>.ome.tif    biref (angle = the POSITIVE angle)
    <base>_<angle><tail>.ome.tif          I(+) positive-angle source
    <base>_-<angle><tail>.ome.tif         I(-) negative-angle source

Recomputation is ONLY possible while the two angle source tiles still exist. Once the
originals are gone (they can be hundreds of GB, so they are often deleted after
stitching) the biref cannot be regenerated -- such files are reported as skipped.

Large images are streamed in row-strips and written via a memory-map when the output
would not fit in the memory budget, so arbitrarily large slides are handled.

Usage:
    ppm-recompute-biref FOLDER [--suffix _colorrms] [--overwrite] [--dry-run]
        [--min-intensity 0] [--strip 2048] [--mem-budget-gb 3]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile as tf

from ppm_library.imaging.writer import TifWriterUtils

# Positive angle sits immediately before "_biref"; tail is whatever follows (e.g. _001).
_BIREF_RE = re.compile(r"^(?P<head>.*_)(?P<angle>-?\d+(?:\.\d+)?)_biref(?P<tail>.*)$")
_EXTS = (".ome.tif", ".ome.tiff", ".tif", ".tiff")


def _split_ext(name):
    for ext in _EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)], name[-len(ext) :]
    return None, None


def sources_for(biref_path: Path):
    """Return (pos_path, neg_path) source candidates for a biref file, or (None, None)."""
    stem, suffix = _split_ext(biref_path.name)
    if stem is None:
        return None, None
    m = _BIREF_RE.match(stem)
    if not m:
        return None, None
    head, angle, tail = m.group("head"), m.group("angle"), m.group("tail")
    neg_angle = angle[1:] if angle.startswith("-") else "-" + angle
    pos_name = f"{head}{angle}{tail}{suffix}"
    neg_name = f"{head}{neg_angle}{tail}{suffix}"
    return biref_path.with_name(pos_name), biref_path.with_name(neg_name)


def output_path(biref_path: Path, suffix: str, overwrite: bool):
    if overwrite:
        return biref_path
    stem, ext = _split_ext(biref_path.name)
    return biref_path.with_name(f"{stem}{suffix}{ext}")


def _read_strip(path, y0, y1, level=0):
    import zarr

    store = tf.imread(str(path), aszarr=True, level=level)
    z = zarr.open(store, mode="r")
    a = np.asarray(z[y0:y1])
    store.close()
    return a.astype(np.float32)[..., :3]


def _pixel_size_um(path):
    try:
        with tf.TiffFile(str(path)) as t:
            om = t.ome_metadata or ""
        m = re.search(r'PhysicalSizeX="([0-9.]+)"', om)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def recompute_one(pos_path, neg_path, out_path, min_intensity, strip, mem_budget_bytes):
    with tf.TiffFile(str(pos_path)) as t:
        h, w = t.series[0].levels[0].shape[:2]
    px = _pixel_size_um(pos_path)
    kw = {}
    if px:
        kw = {
            "metadata": {
                "axes": "YX",
                "PhysicalSizeX": px,
                "PhysicalSizeY": px,
                "PhysicalSizeXUnit": "um",
                "PhysicalSizeYUnit": "um",
            },
            "resolution": (1.0 / px, 1.0 / px),
        }

    def biref_strip(y0, y1):
        p = _read_strip(pos_path, y0, y1)
        n = _read_strip(neg_path, y0, y1)
        return TifWriterUtils.ppm_normalized_difference_abs(p, n, min_intensity=min_intensity)

    if h * w * 2 <= mem_budget_bytes:
        out = np.zeros((h, w), np.uint16)
        for y0 in range(0, h, strip):
            out[y0 : min(y0 + strip, h)] = biref_strip(y0, min(y0 + strip, h))
        tf.imwrite(str(out_path), out, ome=True, tile=(512, 512), compression="zlib", **kw)
    else:
        # Too large to hold in RAM: memory-map a plain OME-TIFF, fill by strips.
        out = tf.memmap(str(out_path), shape=(h, w), dtype=np.uint16, ome=True, **kw)
        for y0 in range(0, h, strip):
            y1 = min(y0 + strip, h)
            out[y0:y1] = biref_strip(y0, y1)
        out.flush()
    return h, w


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folder", help="folder to scan recursively for biref images")
    ap.add_argument(
        "--suffix", default="_colorrms", help="suffix for recomputed output (default _colorrms)"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="overwrite the original biref in place"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would happen, write nothing"
    )
    ap.add_argument(
        "--min-intensity", type=float, default=0.0, help="dark-region mask threshold (8-bit scale)"
    )
    ap.add_argument("--strip", type=int, default=2048, help="row-strip height for streaming")
    ap.add_argument(
        "--mem-budget-gb", type=float, default=3.0, help="max RAM for the output before memmap"
    )
    ap.add_argument(
        "--force", action="store_true", help="recompute even if the output already exists"
    )
    args = ap.parse_args(argv)

    root = Path(args.folder)
    if not root.is_dir():
        print(f"ERROR: not a folder: {root}", file=sys.stderr)
        return 2

    mem = int(args.mem_budget_gb * 1024**3)
    done = missing = skipped = failed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        low = path.name.lower()
        if "biref" not in low or not low.endswith(_EXTS):
            continue
        # Don't reprocess our own outputs.
        if args.suffix and args.suffix.lower() in low:
            continue

        pos, neg = sources_for(path)
        if pos is None:
            print(f"[skip  ] {path.name}: unrecognized biref name pattern")
            skipped += 1
            continue
        if not (pos.exists() and neg.exists()):
            miss = [str(p.name) for p in (pos, neg) if not p.exists()]
            print(
                f"[MISSING] {path.name}: source tiles gone ({', '.join(miss)}) -- cannot recompute"
            )
            missing += 1
            continue

        out = output_path(path, args.suffix, args.overwrite)
        if out.exists() and not (args.overwrite or args.force):
            print(f"[exists] {out.name}: already present (use --force to redo)")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[dry   ] {path.name} -> {out.name}  (pos={pos.name}, neg={neg.name})")
            done += 1
            continue

        try:
            h, w = recompute_one(pos, neg, out, args.min_intensity, args.strip, mem)
            print(f"[ok    ] {out.name}  ({w}x{h})")
            done += 1
        except Exception as e:  # noqa: BLE001 - report and continue the batch
            print(f"[FAIL  ] {path.name}: {e}")
            failed += 1

    print(
        f"\nsummary: {done} recomputed, {missing} un-recomputable (sources gone), "
        f"{skipped} skipped, {failed} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
