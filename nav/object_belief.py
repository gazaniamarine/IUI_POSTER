"""Object-location memory that is *only* the belief: one Gaussian slot per named
object in the real navdp SubgoalBeliefBank (body-frame [forward, left], SE(2)
ego-motion transform + covariance growth while unseen). No map, no BEV, no
landmark graph -- the belief is the memory.

The bank is loaded by file path from navdp/navdp/extensions/belief_bank.py
(the REAL class, never copied) rather than `import navdp.extensions`, whose
package __init__ drags in torch/GroundingDINO modules the habitat env doesn't
need for this.

Pose convention (sam_vla pose_integrator): forward = (-sin yaw, -cos yaw) in
world (x, z); +yaw turns LEFT, left = (-cos yaw, sin yaw).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

_BANK_PATH = (
    Path(__file__).resolve().parent.parent
    / "navdp" / "navdp" / "extensions" / "belief_bank.py"
)


def _load_bank_cls():
    spec = importlib.util.spec_from_file_location("_navdp_belief_bank", _BANK_PATH)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod  # dataclass needs the module registered
    spec.loader.exec_module(mod)
    return mod.SubgoalBeliefBank


def _world_to_body(px, pz, yaw, x, z):
    dx, dz = x - px, z - pz
    return (
        dx * -math.sin(yaw) + dz * -math.cos(yaw),
        dx * -math.cos(yaw) + dz * math.sin(yaw),
    )


def _body_to_world(px, pz, yaw, fwd, left):
    return (
        px + fwd * -math.sin(yaw) + left * -math.cos(yaw),
        pz + fwd * -math.cos(yaw) + left * math.sin(yaw),
    )


def _load_ego():
    import sys

    return sys.modules["_navdp_belief_bank"].ego_motion_update


class ObjectBeliefMemory:
    """Remember where named objects are using SubgoalBeliefBank slots.

    update() once per navigator poll with every object sighted this poll
    (world xz from depth). Between sightings each slot is dead-reckoned by the
    rover's exact pose delta and its sigma grows, so recall() stays correct
    while the rover turns and drives.

    Improvements over a plain "latest sighting wins" bank:
      * repeated sightings are fused by inverse-variance weighting (depth
        error grows with range), so the estimate tightens instead of jumping;
      * a sighting far outside the belief's own 3-sigma gate replaces it
        (the object moved / it is a different instance) instead of blending;
      * negative evidence: note_absent() counts polls where the belief says
        the object is plainly in view but it was not seen -- forget() clears
        the slot so the rover stops trusting a wrong memory."""

    MEAS_SIGMA_BASE = 0.15
    MEAS_SIGMA_PER_M = 0.05
    ABSENT_LIMIT = 3
    CANDIDATE_AGREE_M = 1.0  # a jump is accepted when two sightings agree within this

    def __init__(self, **bank_kwargs):
        bank_kwargs.setdefault("sigma_init", 1.0)
        bank_kwargs.setdefault("sigma_visible", 0.3)
        bank_kwargs.setdefault("odom_noise", 0.02)
        bank_kwargs.setdefault("decay_factor", 0.95)
        self.bank = _load_bank_cls()(goal_ids=[], dim=2, **bank_kwargs)
        self._ego = _load_ego()
        self._last_pose: Optional[tuple] = None
        self._step = 0
        self._absent: dict = {}
        self._cand: dict = {}  # key -> unconfirmed out-of-gate world (x, z)

    def _odom(self, pose):
        if self._last_pose is None:
            return [0.0, 0.0, 0.0]
        px, pz, yaw = self._last_pose
        dfwd, dleft = _world_to_body(px, pz, yaw, pose.x, pose.z)
        dth = (pose.yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        return [dfwd, dleft, dth]

    def update(self, pose, sightings: Mapping[str, Optional[tuple]]) -> None:
        odom = self._odom(pose)
        obs = {}
        sigma_visible_default = self.bank.sigma_visible
        fused_sigma = {}
        for key, world_xz in sightings.items():
            if world_xz is None:
                continue
            # (x, z) from depth, or (x, z, sigma) for a bearing-only sighting
            # placed at an assumed range with a large, explicit sigma
            fwd, left = _world_to_body(pose.x, pose.z, pose.yaw, world_xz[0], world_xz[1])
            rng = math.hypot(fwd, left)
            sig_m = (float(world_xz[2]) if len(world_xz) > 2
                     else self.MEAS_SIGMA_BASE + self.MEAS_SIGMA_PER_M * rng)
            pos = np.array([fwd, left], dtype=np.float32)
            bearing_only = len(world_xz) > 2
            if bearing_only and self.known(key):
                # A bearing-only guess says nothing about range: never let it
                # tighten the belief (repeated guesses were being fused as
                # independent measurements and faking confidence), and never
                # let it override a better estimate.
                if math.sqrt(max(float(self.bank.get(key).Sigma[0, 0]), 0.0)) <= sig_m:
                    self._absent[key] = 0
                    continue
            elif self.known(key):
                slot = self.bank.get(key)
                prior = self._ego(slot.mu, odom)
                sig_p = math.sqrt(max(float(slot.Sigma[0, 0]), 1e-6)) + 0.02
                if float(np.linalg.norm(prior - pos)) < 1.0 + 3.0 * sig_p:
                    wp, wm = 1.0 / sig_p**2, 1.0 / sig_m**2
                    pos = (prior * wp + pos * wm) / (wp + wm)
                    sig_m = math.sqrt(1.0 / (wp + wm))
                    self._cand.pop(key, None)
                else:
                    # Far outside the gate: one bad depth reading (live: a
                    # glass door read at 7 m, 18.6 m, 7 m, 18.9 m -- the depth
                    # camera sees through glass) must not overwrite the memory.
                    # Hold it as a candidate; replace only if the next sighting
                    # agrees with it (within CANDIDATE_AGREE_M) -- a real move.
                    wx, wz = world_xz[0], world_xz[1]
                    cand = self._cand.get(key)
                    if cand is None or math.hypot(cand[0] - wx, cand[1] - wz) > self.CANDIDATE_AGREE_M:
                        self._cand[key] = (wx, wz)
                        self._absent[key] = 0
                        continue
                    self._cand.pop(key, None)  # confirmed: replace with the new position
            fused_sigma[key] = sig_m
            obs[key] = {"visible": True, "position": pos, "confidence": 1.0}
            self._absent[key] = 0
        # bank.sigma_visible is one scalar per bank; apply per-key via a
        # sequence of single-key sightings would double-propagate the others,
        # so use the mean here and then patch each slot's Sigma exactly.
        if fused_sigma:
            self.bank.sigma_visible = float(np.mean(list(fused_sigma.values())))
        self.bank.update(obs, odom, self._step)
        self.bank.sigma_visible = sigma_visible_default
        for key, sig in fused_sigma.items():
            self.bank.get(key).Sigma = np.eye(2, dtype=np.float32) * (sig**2)
        self._last_pose = (pose.x, pose.z, pose.yaw)
        self._step += 1

    def candidate_agrees(self, key: str, world_xz) -> bool:
        """True if an unconfirmed out-of-gate sighting is pending for key and
        world_xz agrees with it (i.e. this sighting would confirm a move)."""
        c = self._cand.get(key)
        return c is not None and math.hypot(c[0] - world_xz[0], c[1] - world_xz[1]) <= self.CANDIDATE_AGREE_M

    def known(self, key: str) -> bool:
        return key in self.bank and self.bank.get(key).initialized

    def keys(self):
        return [k for k in self.bank.goal_ids if self.known(k)]

    def forget(self, key: str) -> None:
        if key in self.bank:
            self.bank.slots[key] = self.bank._new_slot(key)
        self._absent[key] = 0
        self._cand.pop(key, None)

    def note_absent(self, key: str) -> bool:
        """Record 'belief says it is in plain view, but it was not seen'.
        Returns True (and forgets it) once ABSENT_LIMIT polls have agreed."""
        self._absent[key] = self._absent.get(key, 0) + 1
        if self._absent[key] >= self.ABSENT_LIMIT:
            self.forget(key)
            return True
        return False

    def recall(self, key: str, pose) -> Optional[dict]:
        """Where the belief says `key` is *now* (slot transformed from the last
        update pose to `pose`, without mutating it): world xz, bearing
        (rad, +left), distance, 1-sigma (m), visible-at-last-update."""
        if not self.known(key):
            return None
        slot = self.bank.get(key)
        mu = self._ego(slot.mu, self._odom(pose))
        fwd, left = float(mu[0]), float(mu[1])
        return {
            "world_xz": _body_to_world(pose.x, pose.z, pose.yaw, fwd, left),
            "bearing": math.atan2(left, fwd),
            "distance": math.hypot(fwd, left),
            "sigma": float(math.sqrt(max(float(slot.Sigma[0, 0]), 0.0))),
            "visible": bool(slot.visible),
            "time_since_seen": int(slot.time_since_seen),
        }
