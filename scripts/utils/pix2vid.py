"""Compiles a kb_teleop.py recording folder's rgb_*.png / depth_*.png frames
into MP4s.

Usage:
    python scripts/utils/pix2vid.py [--input <mars_teleop_out*>] [--fps 15]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

DEFAULT_FPS = 15
DEFAULT_INPUT_GLOB = "mars_teleop_out*"
RGB_PATTERN = "rgb_*.png"
DEPTH_PATTERN = "depth_*.png"

# Output video names if not provided
RGB_VIDEO_NAME = "rgb_video.mp4"
DEPTH_VIDEO_NAME = "depth_video.mp4"

# ============================================================


def natural_key(path):
    """
    Sort rgb_0001.png, rgb_0002.png, ..., rgb_0010.png correctly.
    """
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def find_latest_recording_folder(pattern):
    folders = [p for p in glob.glob(pattern) if os.path.isdir(p)]

    if not folders:
        raise FileNotFoundError(f"No recording folders found with pattern: {pattern}")

    folders.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return folders[0]


def get_ffmpeg():
    """
    Prefer imageio-ffmpeg from Conda if available.
    Otherwise fall back to system/conda ffmpeg in PATH.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def make_numbered_symlinks(files, temp_dir):
    """
    ffmpeg is happiest with sequential frame_%06d.png input.
    This avoids problems if frames are missing or folder names vary.
    """
    for idx, src in enumerate(files):
        dst = os.path.join(temp_dir, f"frame_{idx:06d}.png")
        os.symlink(os.path.abspath(src), dst)


def frames_to_video(files, out_video, fps):
    if not files:
        print(f"Skipping {out_video}: no frames found.")
        return

    ffmpeg = get_ffmpeg()

    with tempfile.TemporaryDirectory() as temp_dir:
        make_numbered_symlinks(files, temp_dir)

        input_pattern = os.path.join(temp_dir, "frame_%06d.png")

        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            out_video,
        ]

        print("Running:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

    print(f"Wrote: {out_video}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert recorded Habitat RGB-D PNG frames into MP4 videos."
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Recording folder. Example: mars_teleop_out1783002246. "
        "If omitted, latest mars_teleop_out* folder is used.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Output video FPS. Default: {DEFAULT_FPS}",
    )

    parser.add_argument(
        "--rgb-out",
        default=None,
        help="RGB output video path. Default: <input_folder>/rgb_video.mp4",
    )

    parser.add_argument(
        "--depth-out",
        default=None,
        help="Depth output video path. Default: <input_folder>/depth_video.mp4",
    )

    args = parser.parse_args()

    if args.input is None:
        input_dir = find_latest_recording_folder(DEFAULT_INPUT_GLOB)
        print(f"Using latest recording folder: {input_dir}")
    else:
        input_dir = args.input

    input_dir = os.path.abspath(input_dir)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    rgb_files = sorted(
        glob.glob(os.path.join(input_dir, RGB_PATTERN)),
        key=natural_key,
    )

    depth_files = sorted(
        glob.glob(os.path.join(input_dir, DEPTH_PATTERN)),
        key=natural_key,
    )

    print(f"Input folder: {input_dir}")
    print(f"RGB frames:   {len(rgb_files)}")
    print(f"Depth frames: {len(depth_files)}")

    if not rgb_files and not depth_files:
        raise RuntimeError("No RGB or depth frames found.")

    rgb_out = args.rgb_out or os.path.join(input_dir, RGB_VIDEO_NAME)
    depth_out = args.depth_out or os.path.join(input_dir, DEPTH_VIDEO_NAME)

    if rgb_files:
        frames_to_video(rgb_files, rgb_out, args.fps)

    if depth_files:
        frames_to_video(depth_files, depth_out, args.fps)

    print("Ooohh la la!")
    print("RGB video:  ", rgb_out)
    print("Depth video:", depth_out)


if __name__ == "__main__":
    main()
