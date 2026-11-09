"""Ground-truth occupancy from the MuJoCo scene, and metrics to score a BEV map against it."""

import cv2
import mujoco
import numpy as np

from occupancy_grid import FREE, OCCUPIED, UNKNOWN


def ground_truth_grid(model, grid, rover_body, obstacle_min_z, sub=2, return_ids=False):
    """Cast vertical rays over the grid (rover moved out of the way).

    A cell is occupied if anything inside it rises above obstacle_min_z. (The scene has no
    overhangs, so a top-down ray is enough.) return_ids=True also returns, per cell, the id of the
    geom that made it occupied (-1 = free), to tell objects apart.
    """
    d = mujoco.MjData(model)
    rover_jnt = model.body(rover_body).jntadr[0]
    qadr = model.jnt_qposadr[rover_jnt]
    d.qpos[qadr:qadr + 3] = [1000.0, 1000.0, 0.0]
    mujoco.mj_forward(model, d)

    occ = np.zeros((grid.ny, grid.nx), bool)
    ids = np.full((grid.ny, grid.nx), -1, np.int32)
    down = np.array([0.0, 0.0, -1.0])
    top = 5.0
    geomid = np.zeros(1, np.int32)
    offs = (np.arange(sub) + 0.5) / sub * grid.res
    for iy in range(grid.ny):
        y0 = grid.y_min + iy * grid.res
        for ix in range(grid.nx):
            x0 = grid.x_min + ix * grid.res
            hit = False
            for oy in offs:
                for ox in offs:
                    dist = mujoco.mj_ray(model, d, np.array([x0 + ox, y0 + oy, top]), down,
                                         None, 1, -1, geomid)
                    if dist >= 0 and top - dist > obstacle_min_z:
                        hit = True
                        ids[iy, ix] = geomid[0]
                        break
                if hit:
                    break
            occ[iy, ix] = hit
    return (occ, ids) if return_ids else occ


def score(state, gt_occ, region=None, tol_cells=1):
    """Compare a map state (FREE/OCCUPIED/UNKNOWN) against ground-truth occupancy.

    Only cells the map has observed are scored. `tol_cells` allows a small position error
    (one cell = grid resolution) when matching occupied cells.
    """
    known = state != UNKNOWN
    if region is not None:
        known &= region
    m_occ = (state == OCCUPIED) & known
    m_free = (state == FREE) & known
    g_occ = gt_occ & known

    k = np.ones((2 * tol_cells + 1,) * 2, np.uint8)
    gt_dil = cv2.dilate(gt_occ.astype(np.uint8), k) > 0
    m_dil = cv2.dilate(m_occ.astype(np.uint8), k) > 0

    precision = (m_occ & gt_dil).sum() / max(m_occ.sum(), 1)
    recall = (g_occ & m_dil).sum() / max(g_occ.sum(), 1)
    iou = (m_occ & gt_occ).sum() / max((m_occ | g_occ).sum(), 1)
    gt_core = cv2.erode(gt_occ.astype(np.uint8), k) > 0                # ignore 1-cell boundary
    false_free = (m_free & gt_core).sum() / max((gt_core & known).sum(), 1)  # obstacles marked free
    false_occ_cells = int((m_occ & ~gt_dil).sum())                    # "noise" cells
    return {
        "precision": float(precision), "recall": float(recall), "iou_strict": float(iou),
        "false_free": float(false_free), "false_occ_cells": false_occ_cells,
        "known_cells": int(known.sum()), "map_occ_cells": int(m_occ.sum()),
    }
