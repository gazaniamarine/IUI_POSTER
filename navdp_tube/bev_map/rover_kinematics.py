"""Camera pose of the real rover: odometry + IMU + pan/tilt angles -> cam_pos, cam_rot.

The pan-tilt geometry is copied from the rover model (simulation/data/ugv_rover_environment.xml,
generated from the URDF). Output uses the same convention as the simulator, so the BEV pipeline
does not change: world z up, floor z = 0, cam_rot columns = camera x (right), y (up), z (backwards)
in world; the camera looks along -z.

Chain:  world --(odom x, y, yaw + IMU roll, pitch, base_height)--> base
             --(pan joint)--> pt_link1 --(tilt joint)--> pt_link2 --> camera --(mount_rot)--> real camera

The camera sits ~5 cm in front of the tilt axis, so its height changes with tilt (2.4 cm at 30 deg
down). That is why height comes from this chain every frame instead of a single measured number.
mount_rot and base_height come from calibrate_camera.py (checkerboard on the floor).
"""

import numpy as np

# from ugv_rover_environment.xml
P_LINK1 = np.array([-0.00828091, 6.50856e-10, 0.168])       # pt_link1 in rover body
PAN_AXIS = np.array([0.0, 0.0, -1.0])
P_LINK2 = np.array([0.0, 0.02, 0.043003])                    # pt_link2 in pt_link1
Q_LINK2 = (0.707105, 0.707108, 0.0, 0.0)                     # (w, x, y, z)
TILT_AXIS = np.array([0.0, 0.0, 1.0])
P_CAM = np.array([0.047686, -0.000569, 0.02])                # camera in pt_link2
Q_CAM = (0.707107, 0.0, -0.707107, 0.0)
SIM_BASE_HEIGHT = 0.005                                      # rover body origin above floor in sim (settled)


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def rpy_to_mat(roll, pitch, yaw):
    """ROS REP-103 (x forward, y left, z up): R = Rz(yaw) Ry(pitch) Rx(roll). +pitch = nose down."""
    return axis_angle((0, 0, 1), yaw) @ axis_angle((0, 1, 0), pitch) @ axis_angle((1, 0, 0), roll)


def rotvec_to_mat(v):
    v = np.asarray(v, float)
    a = np.linalg.norm(v)
    return np.eye(3) if a < 1e-12 else axis_angle(v / a, a)


def mat_to_rotvec(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))
    return v * a


R_LINK2, R_CAM = quat_to_mat(Q_LINK2), quat_to_mat(Q_CAM)


def camera_in_base(pan, tilt):
    """Camera position (3,) and rotation (3x3) in the rover body frame. pan/tilt = model joint angles (rad)."""
    R1 = axis_angle(PAN_AXIS, pan)
    R2 = R1 @ R_LINK2 @ axis_angle(TILT_AXIS, tilt)
    pos = P_LINK1 + R1 @ (P_LINK2 + R_LINK2 @ axis_angle(TILT_AXIS, tilt) @ P_CAM)
    return pos, R2 @ R_CAM


def camera_pose(x, y, yaw, roll, pitch, pan, tilt, base_height=SIM_BASE_HEIGHT, mount_rot=None):
    """World (odom) camera pose for the pipeline. Angles in rad."""
    Rb = rpy_to_mat(roll, pitch, yaw)
    pc, Rc = camera_in_base(pan, tilt)
    if mount_rot is not None:
        Rc = Rc @ mount_rot                   # small correction fixed to the camera head
    pos = np.array([x, y, base_height]) + Rb @ pc
    return pos, Rb @ Rc


def camera_pitch_deg(cam_rot):
    """How far the optical axis points below the horizon (deg, +ve = down)."""
    fwd = -cam_rot[:, 2]
    return float(np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1))))


class RoverCamera:
    """Telemetry sample -> camera pose, with calibration and sign conventions from the config."""

    def __init__(self, rcfg, calib=None):
        calib = calib or {}
        self.pan_sign = float(rcfg.get("pan_sign", 1.0))
        self.tilt_sign = float(rcfg.get("tilt_sign", 1.0))
        self.roll_sign = float(rcfg.get("imu_roll_sign", 1.0))
        self.pitch_sign = float(rcfg.get("imu_pitch_sign", 1.0))
        self.default_pan, self.default_tilt = rcfg.get("pan_deg", 0.0), rcfg.get("tilt_deg", 0.0)
        mount = calib.get("mount", {})
        self.base_height = float(mount.get("base_height", SIM_BASE_HEIGHT))
        self.mount_rot = rotvec_to_mat(mount.get("rotvec", [0, 0, 0]))
        self.imu_roll0 = float(mount.get("imu_roll_offset", 0.0))
        self.imu_pitch0 = float(mount.get("imu_pitch_offset", 0.0))
        self.calibrated = bool(mount)

    def joint_angles(self, s):
        """Gimbal degrees (as the rover reports them) -> model joint radians."""
        pan = s.get("pan") if s.get("pan") is not None else self.default_pan
        tilt = s.get("tilt") if s.get("tilt") is not None else self.default_tilt
        return self.pan_sign * np.radians(pan), self.tilt_sign * np.radians(tilt)

    def base_tilt(self, s):
        """IMU roll/pitch (rad) with sign fix and the level-floor offset removed."""
        roll = self.roll_sign * s.get("roll", 0.0) - self.imu_roll0
        pitch = self.pitch_sign * s.get("pitch", 0.0) - self.imu_pitch0
        return roll, pitch

    def pose(self, s, use_mount=True):
        roll, pitch = self.base_tilt(s)
        pan, tilt = self.joint_angles(s)
        return camera_pose(s.get("x", 0.0), s.get("y", 0.0), s.get("yaw", 0.0), roll, pitch, pan, tilt,
                           self.base_height, self.mount_rot if use_mount else None)
