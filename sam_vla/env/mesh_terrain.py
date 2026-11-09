"""Navmesh-based terrain sampling, for scenes (e.g. HM3D) that ship a
pre-baked navmesh instead of a DEM heightmap.

Duck-types the same interface `sam_vla.env.terrain.Terrain` exposes
(`__call__(x, z) -> float`, `local_height_max(x, z, radius, samples=5) ->
float`), so `MarsHabitatEnv` can swap terrain backends without changing how
it calls `self._terrain` -- only which one it constructs. See terrain.py's
own module docstring for why marsyard2022 needs a heightmap in the first
place (no queryable ground height from the scene geometry there); HM3D
scenes are scanned meshes that *do* expose one, via their navmesh.
"""

from __future__ import annotations

import logging
import math

import magnum as mn

logger = logging.getLogger(__name__)


# Max plausible single-query vertical jump (metres) before a snap is
# treated as a probable different-floor mistake rather than a real height
# change -- generous enough for stairs/steps/thresholds within one floor
# (rarely more than ~0.3m), well under a typical inter-floor gap (HM3D
# houses commonly run 2.5-3m+ per floor). NOT the same thing as navmesh
# *island* (PathFinder.get_island): islands are disconnected navmesh
# components, which recast/habitat-sim's navmesh baking can also produce
# for two rooms on the *same* floor joined only by a narrow doorway -- an
# island-identity check rejected those legitimate same-floor transitions
# too eagerly when tried live, so this uses height magnitude instead.
_MAX_PLAUSIBLE_STEP_Y = 1.0


class NavmeshTerrain:
    """Samples ground height from a loaded `habitat_sim.nav.PathFinder`'s
    navmesh via `snap_point`, seeded each call with the *previous* result --
    this is what keeps multi-floor scenes on the right floor tick-to-tick
    without needing to change the `Terrain` call signature to carry a Y
    hint (a robot only ever queries height near wherever it already is).

    Also rejects any snap whose height differs from the last known height by
    more than _MAX_PLAUSIBLE_STEP_Y, keeping the last known height instead.
    Without this, a single stray query near a stairwell (where an upper
    floor's navmesh can sit close in (x, z) but far apart in y) can snap to
    the wrong floor and "poison" _last_y for every later call -- confirmed
    live: a query near a staircase mid-hallway snapped to the upper floor,
    after which every subsequent doorway-threshold query (itself failing to
    snap, since navmesh gaps at door thresholds are ordinary) fell back to
    that wrong upper-floor height instead of the correct hallway one,
    floating the rover into the ceiling and stalling it there."""

    def __init__(self, pathfinder, seed_y: float = 0.0):
        if not pathfinder.is_loaded:
            raise ValueError(
                "NavmeshTerrain requires a loaded navmesh (pathfinder.is_loaded "
                "is False) -- load one explicitly or ensure the scene's sibling "
                ".navmesh file auto-loaded"
            )
        self._pathfinder = pathfinder
        self._last_y = float(seed_y)
        self._warned_snap_failure = False
        self._warned_big_jump = False

    def __call__(self, x: float, z: float) -> float:
        hint = mn.Vector3(float(x), self._last_y, float(z))
        snapped = self._pathfinder.snap_point(hint)
        y = float(snapped[1])
        if not math.isfinite(y):
            if not self._warned_snap_failure:
                logger.warning(
                    "NavmeshTerrain.snap_point found no navigable polygon near "
                    "(%.3f, %.3f) -- falling back to last known height %.3f "
                    "(further failures at other points won't log again)",
                    x, z, self._last_y,
                )
                self._warned_snap_failure = True
            return self._last_y

        if abs(y - self._last_y) > _MAX_PLAUSIBLE_STEP_Y:
            if not self._warned_big_jump:
                logger.warning(
                    "NavmeshTerrain.snap_point near (%.3f, %.3f) jumped from "
                    "%.3f to %.3f (> %.1fm) -- likely snapped to a different "
                    "floor; rejecting and keeping last known height (further "
                    "big jumps won't log again)",
                    x, z, self._last_y, y, _MAX_PLAUSIBLE_STEP_Y,
                )
                self._warned_big_jump = True
            return self._last_y

        self._last_y = y
        return self._last_y

    def local_height_max(
        self, x: float, z: float, radius: float, samples: int = 5
    ) -> float:
        """Indoor floors are locally flat (unlike marsyard's rock-strewn
        terrain), so this just delegates to __call__ -- no need for
        Terrain.local_height_max's multi-sample max-within-radius logic,
        which exists there specifically for outdoor bump avoidance."""
        return self(x, z)
