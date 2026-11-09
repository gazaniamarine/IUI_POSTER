"""MuJoCo frame source: RGB, ground-truth depth, intrinsics and exact poses.

Loads the existing simulation XML read-only. Nothing in simulation/ is modified.
"""

import os
from dataclasses import dataclass

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Frame:
    rgb: np.ndarray          # HxWx3 uint8, RGB
    depth: np.ndarray        # HxW float32, metres along the optical axis (MuJoCo ground truth)
    K: np.ndarray            # 3x3 intrinsics
    cam_pos: np.ndarray      # (3,) camera position in world
    cam_rot: np.ndarray      # 3x3, columns = camera x/y/z axes in world (MuJoCo: x right, y up, looks along -z)
    robot_xy: np.ndarray     # (2,) rover position in world
    robot_yaw: float         # rover heading in world (rad)
    time: float
    mask: np.ndarray = None  # HxW bool, False where the pixel shows no geometry (empty void)


def intrinsics_from_fovy(fovy_deg, width, height):
    fy = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
    fx = fy  # MuJoCo uses square pixels
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class SimSource:
    WHEELS = ("left_front_motor", "left_rear_motor", "right_front_motor", "right_rear_motor")

    def __init__(self, cfg, model=None):
        """model: an already compiled MjModel (e.g. the scene with extra obstacles); default: load xml_path."""
        sc = cfg["sim"]
        xml = sc["xml_path"]
        if not os.path.isabs(xml):
            xml = os.path.normpath(os.path.join(HERE, xml))
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.xml_path = xml

        self.width, self.height = sc["render_width"], sc["render_height"]
        self.cam_id = self.model.camera(sc["camera_name"]).id
        self.camera_name = sc["camera_name"]
        self.rover_id = self.model.body(sc["rover_body"]).id
        self.K = intrinsics_from_fovy(self.model.cam_fovy[self.cam_id], self.width, self.height)

        act = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.lf, self.lr, self.rf, self.rr = (act(n) for n in self.WHEELS)
        self.pan, self.tilt = act("camera_pan_motor"), act("camera_tilt_motor")
        self.pan_cmd, self.tilt_cmd = sc.get("camera_pan", 0.0), sc.get("camera_tilt", 0.0)
        self.left_cmd = self.right_cmd = 0.0

        self.mask_background = sc.get("mask_background", True)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- control
    def set_wheels(self, left, right):
        self.left_cmd, self.right_cmd = left, right

    def step(self, n=1):
        d = self.data
        d.ctrl[self.lf] = d.ctrl[self.lr] = self.left_cmd
        d.ctrl[self.rf] = d.ctrl[self.rr] = self.right_cmd
        d.ctrl[self.pan], d.ctrl[self.tilt] = self.pan_cmd, self.tilt_cmd
        for _ in range(n):
            mujoco.mj_step(self.model, d)

    def settle(self, seconds=0.5):
        self.step(int(seconds / self.model.opt.timestep))

    # ---------------------------------------------------------------- sensing
    def robot_pose(self):
        xy = self.data.xpos[self.rover_id][:2].copy()
        R = self.data.xmat[self.rover_id].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return xy, yaw

    def get_frame(self):
        r = self.renderer
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=self.camera_name)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.camera_name)
        depth = r.render().copy().astype(np.float32)
        r.disable_depth_rendering()
        mask = None
        if self.mask_background:
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera=self.camera_name)
            mask = r.render()[..., 0] >= 0
            r.disable_segmentation_rendering()

        xy, yaw = self.robot_pose()
        return Frame(
            rgb=rgb, depth=depth, K=self.K.copy(),
            cam_pos=self.data.cam_xpos[self.cam_id].copy(),
            cam_rot=self.data.cam_xmat[self.cam_id].reshape(3, 3).copy(),
            robot_xy=xy, robot_yaw=yaw, time=float(self.data.time), mask=mask,
        )

    def render_chase(self, distance=1.6, elevation=-30.0):
        """Third-person view following the rover (stand-in for the 3D viewer when headless)."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.rover_id
        cam.distance, cam.elevation = distance, elevation
        _, yaw = self.robot_pose()
        cam.azimuth = float(np.degrees(yaw))
        r = self.renderer
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(self.data, camera=cam)
        return r.render().copy()

    def render_top(self, size=540, center=(0.0, 0.0), distance=5.4):
        """Fixed camera straight above the scene (x right, y up, like the BEV map), size x size pixels."""
        if getattr(self, "top_renderer", None) is None:
            self.top_renderer = mujoco.Renderer(self.model, height=size, width=size)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [center[0], center[1], 0.0]
        cam.elevation, cam.azimuth, cam.distance = -90.0, 90.0, distance
        self.top_renderer.update_scene(self.data, camera=cam)
        return self.top_renderer.render().copy()

    def close(self):
        self.renderer.close()
        if getattr(self, "top_renderer", None) is not None:
            self.top_renderer.close()
