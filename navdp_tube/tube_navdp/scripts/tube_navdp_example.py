#!/usr/bin/env python3
"""The smallest closed loop with TubeDriver (navdp/tube/README.md). The camera is a stand-in and the robot is a perfect
unicycle: replace ``camera`` and the two lines that move the robot with your own simulator or robot. Start
scripts/tube_navdp_server.py first."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navdp.tube.driver import DriverConfig, TubeDriver, WorldGrid  # noqa: E402


def camera(height=240, width=320):
    """Your RGB (uint8) and depth (metres), both [H, W]. Here: a grey ground fading to 5 m."""
    rgb = np.full((height, width, 3), 120, np.uint8)
    depth = np.tile(np.linspace(5.0, 0.5, height, dtype=np.float32)[:, None], (1, width))
    return rgb, depth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:8890")
    p.add_argument("--seconds", type=float, default=2.0)
    args = p.parse_args()
    control_hz, plan_hz = 30, 5
    intrinsic = np.array([[160.0, 0, 159.5], [0, 160.0, 119.5], [0, 0, 1]])  # fx, fy, cx, cy of your camera
    goal = (8.0, 0.0)                     # world x, y in metres
    # Your occupancy map in the world frame: occupied[iy, ix], 5 cm cells from (x_min, y_min) = (-2, -5).
    occupied = np.zeros((200, 260), bool)
    occupied[90:120, 115:145] = True      # a 1.5 m x 1.5 m block at x 3.75..5.25, y -0.5..1.0
    world_map = WorldGrid(occupied, 0.05, (-2.0, -5.0))
    pose = [0.0, 0.0, 0.0]                # x, y, yaw from your odometry

    driver = TubeDriver(args.server, intrinsic, DriverConfig(mode="tube", controller="stt", robot_radius=0.6))
    for step in range(int(args.seconds * control_hz)):
        now = step / control_hz
        if step % (control_hz // plan_hz) == 0:
            rgb, depth = camera()
            result = driver.plan(rgb, depth, goal, pose, world_map, now)
            tubes = result.tube["tubes"]
            print(f"t={now:4.1f} s  candidates {len(result.candidates)}  certified {sum(result.tube['certified'])}"
                  f"  tubes valid {sum(t['feasible'] for t in tubes)}/{len(tubes)}")
        v, omega, info = driver.command(pose, now)
        # Send (v, omega) to your robot here. This stand-in robot does exactly what it is told:
        pose[0] += v * math.cos(pose[2]) / control_hz
        pose[1] += v * math.sin(pose[2]) / control_hz
        pose[2] += omega / control_hz
    print(f"end pose x={pose[0]:.2f} y={pose[1]:.2f} yaw={pose[2]:.2f}; last command v={v:.2f} omega={omega:.2f} {info}")
    driver.close()


if __name__ == "__main__":
    main()
