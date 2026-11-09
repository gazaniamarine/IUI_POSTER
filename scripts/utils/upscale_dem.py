#!/usr/bin/env python3
"""
Stage 1: Upsample the marsyard2022 DEM heightmap from 257x257 to 1025x1025
using bicubic interpolation, for a smooth (non-blocky) result.

Usage:
    python scripts/utils/upscale_dem.py <input.tif> <output.tif> [target_res]
"""

import sys
import numpy as np
import rasterio
from scipy.ndimage import zoom


def upsample_dem(src_path, dst_path, target_res=1025):
    with rasterio.open(src_path) as src:
        data = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        bounds = src.bounds

    assert data.shape[0] == data.shape[1], "Expected square DEM"
    src_res = data.shape[0]

    # keep the classic (2^n + 1) grid convention
    factor = (target_res - 1) / (src_res - 1)

    # order=3 -> bicubic spline: smooth continuous surface, not blocky steps
    upsampled = zoom(data, factor, order=3)

    if upsampled.shape != (target_res, target_res):
        # zoom's output size can be off by a pixel due to float rounding; fix by
        # cropping/padding to the exact target so downstream grid math is exact.
        upsampled = upsampled[:target_res, :target_res]

    new_transform = rasterio.transform.from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, target_res, target_res
    )

    profile.update(
        width=target_res,
        height=target_res,
        transform=new_transform,
        dtype="float64",
    )

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(upsampled, 1)

    print(f"source:    {src_res}x{src_res}  min={data.min():.4f} max={data.max():.4f}")
    print(
        f"upsampled: {upsampled.shape[0]}x{upsampled.shape[1]}  "
        f"min={upsampled.min():.4f} max={upsampled.max():.4f}"
    )
    print(f"bounds: {bounds}")
    print(f"saved -> {dst_path}")

    return upsampled, bounds


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 upsample_dem.py <input.tif> <output.tif> [target_res]")
        sys.exit(1)

    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    target_res = int(sys.argv[3]) if len(sys.argv) > 3 else 1025

    upsample_dem(src_path, dst_path, target_res)
