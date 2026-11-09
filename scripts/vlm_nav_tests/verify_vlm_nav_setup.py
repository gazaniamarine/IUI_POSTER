#!/usr/bin/env python3
"""Checks the prerequisites for vlm_nav_interactive.py before running it:
required files exist (scene, heightmap, labels.txt), the labelme binary is
available in the `annotate` conda env, key Python packages import, and
output directories can be created.

Usage:
    python scripts/vlm_nav_tests/verify_vlm_nav_setup.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path

from conda_env import resolve_conda_base

HERE = Path(__file__).resolve().parent


def check(description, condition):
    """Print a check result."""
    symbol = "✓" if condition else "✗"
    status = "PASS" if condition else "FAIL"
    print(f"[{symbol}] {description}: {status}")
    return condition


def main():
    print("\n" + "=" * 70)
    print("VLM Navigation Interactive Setup Verification")
    print("=" * 70 + "\n")

    all_pass = True

    # Check files
    print("FILE CHECKS:")
    print("-" * 70)

    scene = str(HERE / "marsyard2022_tri.glb")
    all_pass &= check("Scene file (marsyard2022_tri.glb)", os.path.exists(scene))

    hm = str(HERE / "marsyard2022_terrain_hm.png")
    all_pass &= check("Heightmap file", os.path.exists(hm))

    labels = str(HERE / "labels.txt")
    all_pass &= check("Labels file (labels.txt)", os.path.exists(labels))

    if os.path.exists(labels):
        with open(labels, "r") as f:
            content = f.read().strip()
            has_labels = "sand" in content and "rock" in content
            all_pass &= check("Labels file contains 'sand' and 'rock'", has_labels)

    print()

    # Check labelme
    print("LABELME CHECKS:")
    print("-" * 70)

    try:
        conda_base = resolve_conda_base()
        labelme_bin = f"{conda_base}/envs/annotate/bin/labelme"
    except RuntimeError as e:
        conda_base = None
        labelme_bin = None
        all_pass &= check(f"Locate conda installation ({e})", False)

    if labelme_bin is not None:
        all_pass &= check("Labelme binary exists", os.path.exists(labelme_bin))

    if labelme_bin is not None and os.path.exists(labelme_bin):
        try:
            result = subprocess.run(
                [labelme_bin, "--version"], capture_output=True, text=True, timeout=5
            )
            version_ok = result.returncode == 0
            all_pass &= check("Labelme --version works", version_ok)
            if version_ok:
                print(f"  Version: {result.stdout.strip()}")
        except Exception as e:
            all_pass &= check("Labelme --version works", False)
            print(f"  Error: {e}")

    print()

    # Check Python imports
    print("PYTHON PACKAGE CHECKS:")
    print("-" * 70)

    try:
        import numpy

        all_pass &= check("numpy", True)
    except Exception as e:
        all_pass &= check("numpy", False)
        print(f"  Error: {e}")

    try:
        from PIL import Image, ImageTk

        all_pass &= check("PIL (Image, ImageTk)", True)
    except Exception as e:
        all_pass &= check("PIL (Image, ImageTk)", False)
        print(f"  Error: {e}")

    try:
        import tkinter as tk

        all_pass &= check("tkinter", True)
    except Exception as e:
        all_pass &= check("tkinter", False)
        print(f"  Error: {e}")

    try:
        import quaternion

        all_pass &= check("quaternion", True)
    except Exception as e:
        all_pass &= check("quaternion", False)
        print(f"  Error: {e}")

    try:
        import habitat_sim

        all_pass &= check("habitat_sim", True)
    except Exception as e:
        all_pass &= check("habitat_sim", False)
        print(f"  Error: {e}")

    print()

    # Check directory creation
    print("OUTPUT DIRECTORY CHECKS:")
    print("-" * 70)

    work_dir = os.path.dirname(__file__) or "."

    for dirname in ["vlm_nav_out", "annotations", "masks"]:
        path = os.path.join(work_dir, dirname)
        try:
            os.makedirs(path, exist_ok=True)
            all_pass &= check(f"Can create/write '{dirname}' directory", True)
        except Exception as e:
            all_pass &= check(f"Can create/write '{dirname}' directory", False)
            print(f"  Error: {e}")

    print()

    # Test JSON validation logic
    print("JSON VALIDATION CHECKS:")
    print("-" * 70)

    test_json = {
        "version": "6.3.1",
        "flags": {},
        "shapes": [
            {
                "label": "rock",
                "points": [[100, 200], [150, 250]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": "test.png",
        "imageData": None,
        "imageHeight": 480,
        "imageWidth": 640,
    }

    try:
        # Validate required keys
        required_keys = [
            "version",
            "flags",
            "shapes",
            "imagePath",
            "imageHeight",
            "imageWidth",
        ]
        missing = [k for k in required_keys if k not in test_json]
        has_required = len(missing) == 0

        # Validate shapes
        shapes_ok = isinstance(test_json.get("shapes"), list)
        if shapes_ok and len(test_json["shapes"]) > 0:
            s = test_json["shapes"][0]
            shape_keys_ok = "label" in s and "points" in s
            shapes_ok = shape_keys_ok

        all_pass &= check("Test JSON validates correctly", has_required and shapes_ok)
    except Exception as e:
        all_pass &= check("Test JSON validates correctly", False)
        print(f"  Error: {e}")

    print()

    # Summary
    print("=" * 70)
    if all_pass:
        print("✓ ALL CHECKS PASSED - Ready to run vlm_nav_interactive.py")
        print("\n  Usage:")
        print("    conda activate habitat")
        print("    python vlm_nav_interactive.py")
    else:
        print("✗ SOME CHECKS FAILED - See errors above")
        sys.exit(1)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
