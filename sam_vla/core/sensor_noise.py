"""Study 2 (next.md): ports belief_exp/scenario.py's odometry-noise formula
(env_odom_noise_std, scenario.py:144-151) into a real Habitat-Sim rollout's
actual driving deltas. Per belief_exp's own "never copies or reimplements the
real classes" discipline and the ghost-mask precedent ("port the pattern, not
the import"), this is a new sam_vla/ module rather than an import of
belief_exp/scenario.py -- belief_exp stays a read-only offline harness.

Pure function, no sim dependency -- same style as core/uncertainty_motion.py's
primitives.
"""

from __future__ import annotations

import numpy as np

from sam_vla.core.types import Action


def apply_odom_noise(
    action: Action, odom_noise_std: float, rng: np.random.Generator
) -> Action:
    """Returns a new Action with Gaussian noise added to the commanded
    velocities, matching belief_exp/scenario.py's split exactly: full-scale
    noise (std=odom_noise_std) on the translational components (v_fwd,
    v_lat), half-scale (std=odom_noise_std * 0.5) on yaw_rate -- the same
    dx/dy/dtheta split scenario.py applies, since these are the same
    physical quantities pre-integration (integrate_mars multiplies each by
    dt to get the actual delta).

    odom_noise_std <= 0.0 returns action unchanged (no RNG draw), so a
    disabled --drive-odom-noise-std leaves behavior byte-identical to before
    this existed.
    """
    if odom_noise_std <= 0.0:
        return action
    odom_noise_xy = odom_noise_std
    odom_noise_th = odom_noise_std * 0.5
    return Action(
        v_fwd=action.v_fwd + float(rng.normal(0.0, odom_noise_xy)),
        v_lat=action.v_lat + float(rng.normal(0.0, odom_noise_xy)),
        yaw_rate=action.yaw_rate + float(rng.normal(0.0, odom_noise_th)),
    )
