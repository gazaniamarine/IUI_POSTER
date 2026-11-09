#!/usr/bin/env python3
"""Calibrate the real rover camera with a printed checkerboard. Writes calib/real_camera.yaml.

    # 1. intrinsics + lens distortion: hold the board in front of the camera at many angles/distances
    python calibrate_camera.py capture --out calib/images          # saves ~25 good views automatically
    python calibrate_camera.py intrinsics --images calib/images

    # 2. mount: board lying FLAT on the floor 0.4-0.9 m in front of the rover, rover standing still
    #    (rover_state_bridge.py running, so IMU and gimbal angles are recorded too)
    python calibrate_camera.py mount

    # 3. check: move the board / tilt the gimbal; measured and predicted camera pose should agree
    python calibrate_camera.py check

--board is the number of INNER corners (columns x rows), --square the square size in metres.

What the mount step measures and why: the board gives the true camera height above the floor and the
true direction of "up" as the camera sees it. The rover model (rover_kinematics.py) predicts both
from the gimbal angles. The difference is stored as a small mount rotation + base height, so the
pipeline's floor model (camera height + pitch) matches the real rover. The IMU reading at this
moment is stored as its level-floor offset.
"""

import argparse
import os
import time

import cv2
import numpy as np

from bev_pipeline import load_config
from projection import _rot_between
from real_source import LiveSource, load_calibration, save_calibration
from rover_kinematics import RoverCamera, camera_in_base, camera_pitch_deg, mat_to_rotvec

HEADLESS = not os.environ.get("DISPLAY")
CV2MJ = np.diag([1.0, -1.0, -1.0])            # OpenCV camera axes -> pipeline (MuJoCo) camera axes


def board_points(cols, rows, square):
    obj = np.zeros((rows * cols, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def find_board(gray, cols, rows):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, c, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))


def make_display(args):
    from web_view import Display
    d = Display("calibration", web=args.web or HEADLESS, port=args.port)
    if d.url:
        print(f"Open {d.url} to see the camera")
    return d


def raw_camera(cfg):
    from gstream import CameraStream
    return CameraStream(port=cfg["real"].get("camera_port", 5000))


# ---------------------------------------------------------------- capture
def cmd_capture(args, cfg):
    cols, rows = args.board
    os.makedirs(args.out, exist_ok=True)
    cam, disp = raw_camera(cfg), make_display(args)
    saved, last_sig, last_t = 0, None, 0.0
    print(f"Move the board around the whole image, tilted in different directions. Need {args.count}.")
    try:
        while saved < args.count:
            bgr = cam.read()
            if bgr is None:
                time.sleep(0.01)
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            c = find_board(gray, cols, rows)
            vis = bgr.copy()
            if c is not None:
                cv2.drawChessboardCorners(vis, (cols, rows), c, True)
                h, w = gray.shape
                pts = c.reshape(-1, 2)
                sig = np.r_[pts.mean(0) / [w, h], (pts.max(0) - pts.min(0)) / [w, h]]
                new_view = last_sig is None or np.abs(sig - last_sig).max() > 0.08
                if new_view and time.time() - last_t > args.min_interval:
                    cv2.imwrite(os.path.join(args.out, f"view_{saved:03d}.png"), bgr)
                    saved, last_sig, last_t = saved + 1, sig, time.time()
                    print(f"  saved {saved}/{args.count}")
            cv2.putText(vis, f"saved {saved}/{args.count}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            disp.show(vis)
            if "q" in disp.keys():
                break
    finally:
        cam.release()
        disp.close()


# ---------------------------------------------------------------- intrinsics
def cmd_intrinsics(args, cfg):
    cols, rows = args.board
    obj = board_points(cols, rows, args.square)
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".png", ".jpg")))
    objs, imgs, size = [], [], None
    for f in files:
        gray = cv2.imread(os.path.join(args.images, f), cv2.IMREAD_GRAYSCALE)
        size = gray.shape[::-1]
        c = find_board(gray, cols, rows)
        if c is not None:
            objs.append(obj)
            imgs.append(c)
    print(f"board found in {len(imgs)}/{len(files)} images")
    if len(imgs) < 8:
        raise SystemExit("need at least 8 good views")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    print(f"reprojection error {rms:.3f} px (good: < 0.5)\nK =\n{K}\ndist = {dist.ravel()}")
    hfov = 2 * np.degrees(np.arctan(size[0] / 2 / K[0, 0]))
    print(f"horizontal FOV {hfov:.1f} deg")
    calib = load_calibration(cfg["real"])
    calib["camera"] = {"width": int(size[0]), "height": int(size[1]), "K": K.tolist(),
                       "dist": dist.ravel().tolist(), "rms": float(rms), "hfov_deg": float(hfov),
                       "date": time.strftime("%Y-%m-%d %H:%M")}
    print(f"wrote {save_calibration(cfg['real'], calib)}")


# ---------------------------------------------------------------- mount / check
def measure_board(rgb, K, cols, rows, square, thickness):
    """Camera height above the floor and 'up' in the pipeline camera frame, from a board on the floor."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    c = find_board(gray, cols, rows)
    if c is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(board_points(cols, rows, square), c, K, None)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    n, t = R[:, 2], tvec.ravel()               # board normal and board origin, OpenCV camera frame
    if n @ (-t) < 0:                           # make the normal point towards the camera (= up)
        n = -n
    height = abs(n @ t) + thickness
    return {"height": height, "up": CV2MJ @ n, "corners": c}


def up_to_pitch_roll(up):
    """Camera pitch (deg, +down) and roll (deg) from 'up' in the pipeline camera frame."""
    pitch = np.degrees(np.arctan2(up[2], up[1]))     # optical axis is -z: up tilting to +z = looking down
    roll = np.degrees(np.arctan2(-up[0], up[1]))
    return pitch, roll


def cmd_mount(args, cfg, check=False):
    cols, rows = args.board
    rc = cfg["real"]
    calib = load_calibration(rc)
    if "camera" not in calib:
        print("[warn] no intrinsics yet - run 'intrinsics' first for accurate results")
    src = LiveSource(cfg, calib, telemetry=not args.no_telemetry)
    disp = make_display(args)
    cam_uncal = RoverCamera(rc, {})            # model only: no mount correction, no IMU offset
    cam_cal = RoverCamera(rc, calib)
    hs, ups, imu, gimbal = [], [], [], []
    print("Board flat on the floor, rover still." if not check else
          "Checking: compare 'measured' with 'predicted (calibrated)'. Q to quit.")
    try:
        while check or len(hs) < args.frames:
            item = src.next()
            if item is None:
                print("no camera frames")
                continue
            rgb, K, _, s = item
            s = s or {}
            m = measure_board(rgb, K, cols, rows, args.square, args.board_thickness)
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if m is not None:
                cv2.drawChessboardCorners(vis, (cols, rows), m["corners"], True)
                p, r = up_to_pitch_roll(m["up"])
                txt = f"measured: h={m['height'] * 100:.1f}cm pitch={p:+.1f} roll={r:+.1f}"
                cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if check:
                    pos, R = cam_cal.pose(s)
                    pc, rc_ = up_to_pitch_roll(R.T @ np.array([0, 0, 1.0]))
                    pred = f"predicted: h={pos[2] * 100:.1f}cm pitch={pc:+.1f} roll={rc_:+.1f}"
                    cv2.putText(vis, pred, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    print(f"\r{txt} | {pred} | gimbal pan={s.get('pan')} tilt={s.get('tilt')}   ", end="")
                else:
                    hs.append(m["height"]); ups.append(m["up"])
                    imu.append((s.get("roll") or 0.0, s.get("pitch") or 0.0))
                    gimbal.append(cam_uncal.joint_angles(s))
                    print(f"\r  {len(hs)}/{args.frames}  {txt}   ", end="")
            disp.show(vis)
            if "q" in disp.keys():
                break
    finally:
        src.close()
        disp.close()
    print()
    if check or len(hs) < args.frames:
        return

    h_meas = float(np.median(hs))
    up = np.median(np.array(ups), 0)
    up /= np.linalg.norm(up)
    pan, tilt = np.median(np.array(gimbal), 0)
    _, Rbc = camera_in_base(pan, tilt)
    pc, _ = camera_in_base(pan, tilt)
    up_pred = Rbc.T @ np.array([0, 0, 1.0])    # base level (the floor is level, IMU offset removed)
    M = _rot_between(up, up_pred)              # mount correction: makes the model see the measured 'up'
    corr_deg = float(np.degrees(np.linalg.norm(mat_to_rotvec(M))))
    roll0, pitch0 = np.median(np.array(imu), 0)
    mount = {
        "base_height": float(h_meas - pc[2]),
        "rotvec": mat_to_rotvec(M).tolist(),
        "imu_roll_offset": float(cam_uncal.roll_sign * roll0),
        "imu_pitch_offset": float(cam_uncal.pitch_sign * pitch0),
        "measured_cam_height": h_meas,
        "measured_pitch_deg": float(up_to_pitch_roll(up)[0]),
        "measured_roll_deg": float(up_to_pitch_roll(up)[1]),
        "model_pitch_deg": float(up_to_pitch_roll(up_pred)[0]),
        "correction_deg": corr_deg,
        "height_spread_cm": float(np.std(hs) * 100),
        "gimbal_pan_tilt_rad": [float(pan), float(tilt)],
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(f"camera height {h_meas * 100:.1f} cm (spread {mount['height_spread_cm']:.2f} cm), "
          f"pitch {mount['measured_pitch_deg']:+.2f} deg (model said {mount['model_pitch_deg']:+.2f}), "
          f"roll {mount['measured_roll_deg']:+.2f} deg")
    print(f"mount correction {corr_deg:.2f} deg; IMU offset roll {np.degrees(roll0):+.2f} "
          f"pitch {np.degrees(pitch0):+.2f} deg")
    if corr_deg > 8:
        print("[warn] correction > 8 deg: check real.tilt_sign / pan_sign and the gimbal angles")
    calib["mount"] = mount
    print(f"wrote {save_calibration(rc, calib)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["capture", "intrinsics", "mount", "check"])
    ap.add_argument("--board", default="9x6", help="inner corners, columns x rows")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    ap.add_argument("--board-thickness", type=float, default=0.0, help="m, board above the floor (mount)")
    ap.add_argument("--out", default="calib/images")
    ap.add_argument("--images", default="calib/images")
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--min-interval", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=30, help="frames averaged for the mount")
    ap.add_argument("--no-telemetry", action="store_true", help="mount without IMU/gimbal data")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    args.board = tuple(int(v) for v in args.board.lower().split("x"))
    cfg = load_config(overrides=args.set)
    {"capture": cmd_capture, "intrinsics": cmd_intrinsics,
     "mount": cmd_mount, "check": lambda a, c: cmd_mount(a, c, check=True)}[args.step](args, cfg)


if __name__ == "__main__":
    main()
