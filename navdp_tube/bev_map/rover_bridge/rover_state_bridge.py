#!/usr/bin/env python3
"""ROVER SIDE: send odometry + IMU (+ gimbal angles) to the GPU server for BEV mapping.

Runs inside the rover's ROS 2 container, next to navdp_cmdvel_bridge.py:

    docker exec -it ugv_rpi_ros_humble bash
    python3 rover_state_bridge.py --server <SERVER_IP>
    # check topic names first:  ros2 topic list ; ros2 topic echo /odom --once ; ros2 topic echo /imu/data --once

Publishes JSON over ZMQ (connects to tcp://<SERVER_IP>:5557, the server binds), ~30 Hz:
    {"t", "x", "y", "yaw", "vx", "wz", "roll", "pitch", "gx", "gy", "gz", "pan", "tilt"}
Positions in metres (odom frame), angles in rad, except pan/tilt in degrees as the gimbal reports
them (null if no gimbal topic). Missing sources are sent as null and the server falls back.

Only needs rclpy, nav_msgs, sensor_msgs and pyzmq (already used by the cmd_vel bridge).
"""

import argparse
import json
import math
import time

import rclpy
import zmq
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState


def quat_to_rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class StateBridge(Node):
    def __init__(self, a):
        super().__init__("bev_state_bridge")
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 10)
        self.sock.connect(f"tcp://{a.server}:{a.port}")
        self.state = {k: None for k in ("x", "y", "yaw", "vx", "wz", "roll", "pitch", "gx", "gy", "gz",
                                        "pan", "tilt")}
        self.pan_name, self.tilt_name = a.pan_joint, a.tilt_joint
        self.n_odom = self.n_imu = 0
        self.create_subscription(Odometry, a.odom_topic, self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Imu, a.imu_topic, self.on_imu, qos_profile_sensor_data)
        if a.gimbal_topic:
            self.create_subscription(JointState, a.gimbal_topic, self.on_gimbal, 10)
        self.create_timer(1.0 / a.hz, self.send)
        self.create_timer(5.0, self.report)
        self.get_logger().info(f"odom={a.odom_topic} imu={a.imu_topic} gimbal={a.gimbal_topic or '-'} "
                               f"-> tcp://{a.server}:{a.port}")

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        _, _, yaw = quat_to_rpy(q)
        self.state.update(x=p.x, y=p.y, yaw=yaw, vx=m.twist.twist.linear.x, wz=m.twist.twist.angular.z)
        self.n_odom += 1

    def on_imu(self, m):
        roll, pitch, _ = quat_to_rpy(m.orientation)
        g = m.angular_velocity
        self.state.update(roll=roll, pitch=pitch, gx=g.x, gy=g.y, gz=g.z)
        self.n_imu += 1

    def on_gimbal(self, m):
        for name, pos in zip(m.name, m.position):
            if name == self.pan_name:
                self.state["pan"] = math.degrees(pos)
            elif name == self.tilt_name:
                self.state["tilt"] = math.degrees(pos)

    def send(self):
        msg = dict(self.state, t=time.time())
        try:
            self.sock.send_string(json.dumps(msg), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def report(self):
        self.get_logger().info(f"last 5 s: odom {self.n_odom} msgs, imu {self.n_imu} msgs")
        if self.n_odom == 0 or self.n_imu == 0:
            self.get_logger().warn("no odom or no imu messages - check topic names (ros2 topic list)")
        self.n_odom = self.n_imu = 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="IP of the GPU server running run_bev_real.py")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--gimbal-topic", default="", help="sensor_msgs/JointState with pan/tilt (optional)")
    ap.add_argument("--pan-joint", default="pt_base_link_to_pt_link1")
    ap.add_argument("--tilt-joint", default="pt_link1_to_pt_link2")
    ap.add_argument("--hz", type=float, default=30.0)
    a = ap.parse_args()
    rclpy.init()
    node = StateBridge(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close(0)
        node.ctx.term()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
