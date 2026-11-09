"""BEV occupancy grid -> the occupied cells the tube planner uses (tube_nav.nav_map wraps them as a WorldGrid).

  1. erase a disk around the goal (goal_clear_radius): the target object must not be an obstacle, or the last
     waypoints sit in occupied cells and no tube reaches the goal;
  2. keep cells within obstacle_range of the robot;
  3. drop specks smaller than min_blob_cells.

The cells do NOT include the robot's size: the tube driver grows them by robot_radius itself. Do not inflate the
grid before calling this, or obstacles grow twice.
"""

import cv2
import numpy as np

from occupancy_grid import OCCUPIED


def goal_clear_mask(grid, goal_xy, radius):
    """Cells whose centre lies within `radius` of the goal."""
    if goal_xy is None or radius <= 0:
        return np.zeros(grid.L.shape, bool)
    return np.hypot(grid.cell_x - goal_xy[0], grid.cell_y - goal_xy[1]) <= radius


def obstacle_cells(grid, state, robot_xy, goal_xy, nc):
    """Occupied cells used for planning: goal disk erased, within range, specks removed."""
    occ = state == OCCUPIED
    occ &= ~goal_clear_mask(grid, goal_xy, nc.get("goal_clear_radius", 0.4))
    occ &= np.hypot(grid.cell_x - robot_xy[0], grid.cell_y - robot_xy[1]) <= nc.get("obstacle_range", 2.5)
    min_cells = int(nc.get("min_blob_cells", 4))
    if min_cells > 1 and occ.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(occ.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_cells
        occ = keep[lab]
    return occ
