"""Tests for the ppm-recompute-biref pyramidal output tool."""

import numpy as np
import tifffile as tf

from ppm_library.imaging.writer import TifWriterUtils
from ppm_library.tools import recompute_biref as rb


def test_pyramid_downsamples_matches_factor2_stop_below_tile():
    # 2000x2000, tile 512: 2000, 1000, (500 < 512 -> stop) => [1, 2]
    assert rb.pyramid_downsamples(2000, 2000, 512) == [1, 2]
    # 200x200, tile 32: 200, 100, 50, (25 < 32 -> stop) => [1, 2, 4]
    assert rb.pyramid_downsamples(200, 200, 32) == [1, 2, 4]
    # capped at 16 levels for a huge image (100M px would otherwise yield 18)
    assert len(rb.pyramid_downsamples(100_000_000, 100_000_000, 512)) == 16


def test_downsample_half_area_average():
    a = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], np.uint16)
    out = rb.downsample_half(a, strip=8)
    assert out.shape == (1, 2)
    assert out.tolist() == [[0, 100]]


def _write_rgb(path, arr, px):
    # OME-TIFF RGB source with an explicit micrometer physical size so
    # _pixel_size_um can recover it from the OME-XML.
    tf.imwrite(
        str(path),
        arr,
        photometric="rgb",
        ome=True,
        metadata={
            "axes": "YXC",
            "PhysicalSizeX": px,
            "PhysicalSizeY": px,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeYUnit": "um",
        },
    )


def test_recompute_one_pyramidal_and_pixel_exact(tmp_path):
    rng = np.random.default_rng(0)
    h, w = 200, 200
    px = 0.37
    pos = rng.integers(20, 255, size=(h, w, 3), dtype=np.uint8)
    neg = rng.integers(20, 255, size=(h, w, 3), dtype=np.uint8)

    pos_path = tmp_path / "slideA_7_001.ome.tif"
    neg_path = tmp_path / "slideA_-7_001.ome.tif"
    _write_rgb(pos_path, pos, px)
    _write_rgb(neg_path, neg, px)

    out = tmp_path / "out.ome.tif"
    rb.recompute_one(
        pos_path,
        neg_path,
        out,
        min_intensity=0.0,
        strip=64,
        tile=32,
        compression="lzw",
        mem_bytes=10 * 1024**3,
    )

    with tf.TiffFile(str(out)) as t:
        series = t.series[0]
        levels = series.levels
        # tile 32 on 200x200 -> [1, 2, 4] => 3 pyramid levels
        assert len(levels) == 3
        level0 = levels[0].asarray()
        assert level0.dtype == np.uint16
        assert level0.shape == (h, w)
        # level dimensions halve
        assert levels[1].shape[:2] == (h // 2, w // 2)
        assert levels[2].shape[:2] == (h // 4, w // 4)
        # tiled 512? we asked for 32
        page = levels[0].pages[0]
        assert page.tilewidth == 32 and page.tilelength == 32
        assert page.compression == tf.COMPRESSION.LZW
        # physical size carried over
        om = t.ome_metadata or ""
        assert 'PhysicalSizeX="0.37"' in om

    # level 0 must equal the direct per-channel biref, bit-for-bit (lossless).
    expected = TifWriterUtils.ppm_normalized_difference_abs(
        pos.astype(np.float32), neg.astype(np.float32), min_intensity=0.0
    )
    assert np.array_equal(level0, expected)


def test_recompute_one_memmap_path_matches_ram_path(tmp_path):
    # Force the memmap branch (mem_bytes=0) and confirm identical level-0 output.
    rng = np.random.default_rng(1)
    h, w = 128, 96
    px = 0.5
    pos = rng.integers(20, 255, size=(h, w, 3), dtype=np.uint8)
    neg = rng.integers(20, 255, size=(h, w, 3), dtype=np.uint8)
    pos_path = tmp_path / "s_5_001.ome.tif"
    neg_path = tmp_path / "s_-5_001.ome.tif"
    _write_rgb(pos_path, pos, px)
    _write_rgb(neg_path, neg, px)

    out = tmp_path / "out.ome.tif"
    rb.recompute_one(
        pos_path,
        neg_path,
        out,
        min_intensity=0.0,
        strip=32,
        tile=32,
        compression="lzw",
        mem_bytes=0,
    )
    # temp memmap file must be cleaned up
    assert not (out.with_name(out.name + ".basetmp.dat")).exists()

    with tf.TiffFile(str(out)) as t:
        level0 = t.series[0].levels[0].asarray()
    expected = TifWriterUtils.ppm_normalized_difference_abs(
        pos.astype(np.float32), neg.astype(np.float32), min_intensity=0.0
    )
    assert np.array_equal(level0, expected)


def test_sources_for_naming():
    p = rb.sources_for(rb.Path("dir/slideA_7_biref_001.ome.tif"))
    assert p[0].name == "slideA_7_001.ome.tif"
    assert p[1].name == "slideA_-7_001.ome.tif"
