"""Drawing helpers for the BEV map. Global view: world +x to the right, +y up."""

import cv2
import numpy as np

from occupancy_grid import FREE, OCCUPIED, UNKNOWN

COLORS = {  # BGR
    UNKNOWN: (110, 110, 110),
    FREE: (235, 235, 235),
    OCCUPIED: (30, 30, 30),
}
INFLATED = (150, 190, 235)
GT_EDGE = (40, 40, 220)
ROBOT = (200, 120, 0)
FOV = (60, 170, 60)


def world_to_px(grid, scale, x, y):
    px = (x - grid.x_min) / grid.res * scale
    py = (grid.y_max - y) / grid.res * scale
    return int(round(px)), int(round(py))


def render_global(grid, state, scale, robot_xy=None, robot_yaw=None, robot_radius=0.13,
                  inflated=None, gt_occ=None, fov=None, max_range=None, trail=None):
    img = np.empty(state.shape + (3,), np.uint8)
    for k, c in COLORS.items():
        img[state == k] = c
    if inflated is not None:
        img[inflated & (state != OCCUPIED)] = INFLATED
    img = np.flipud(img)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    if gt_occ is not None:
        edge = np.flipud(gt_occ.astype(np.uint8))
        edge = cv2.resize(edge, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, GT_EDGE, 1)

    if trail is not None and len(trail) > 1:
        pts = np.array([world_to_px(grid, scale, x, y) for x, y in trail], np.int32)
        cv2.polylines(img, [pts], False, (0, 160, 255), 1, cv2.LINE_AA)

    if robot_xy is not None:
        draw_robot(img, grid, scale, robot_xy, robot_yaw, robot_radius, fov, max_range)
    return img


def draw_robot(img, grid, scale, xy, yaw, radius, fov=None, max_range=None):
    c = world_to_px(grid, scale, *xy)
    if fov is not None and max_range is not None:
        for s in (-1, 1):
            a = yaw + s * fov / 2
            e = world_to_px(grid, scale, xy[0] + max_range * np.cos(a), xy[1] + max_range * np.sin(a))
            cv2.line(img, c, e, FOV, 1, cv2.LINE_AA)
    r = max(2, int(radius / grid.res * scale))
    cv2.circle(img, c, r, ROBOT, 2, cv2.LINE_AA)
    tip = world_to_px(grid, scale, xy[0] + 1.6 * radius * np.cos(yaw), xy[1] + 1.6 * radius * np.sin(yaw))
    cv2.arrowedLine(img, c, tip, ROBOT, 2, cv2.LINE_AA, tipLength=0.4)


def render_local(global_img, grid, scale, robot_xy, robot_yaw, size_m):
    """Robot-centred crop of the global image, rotated so the robot faces up."""
    c = world_to_px(grid, scale, *robot_xy)
    # heading is drawn at `yaw` CCW from image-right; rotate the image CCW by (90 - yaw) so it points up
    M = cv2.getRotationMatrix2D(c, 90.0 - np.degrees(robot_yaw), 1.0)
    side = int(size_m / grid.res * scale)
    M[0, 2] += side / 2 - c[0]
    M[1, 2] += side * 0.65 - c[1]          # robot sits below centre so more of the front is shown
    return cv2.warpAffine(global_img, M, (side, side), flags=cv2.INTER_NEAREST,
                          borderValue=COLORS[UNKNOWN])


def depth_vis(depth, max_d=4.0):
    d = np.nan_to_num(depth, nan=0, posinf=0)
    v = (255 * np.clip(1 - d / max_d, 0, 1)).astype(np.uint8)
    out = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
    out[d <= 0] = 0
    return out


def label(img, text, org=(8, 22), color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return img
