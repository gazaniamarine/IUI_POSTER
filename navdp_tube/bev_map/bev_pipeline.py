"""Frame -> depth -> filtered 3D points -> fused occupancy grid.

The ground is the plane z = 0 unless the frame also gives floor_depth (HxW depth of the known terrain at each pixel)
and ground_z(x, y) (terrain height): then heights are measured above the terrain (e.g. the Habitat Mars yard).
sam2 also takes frame.mask (pixels that are part of the scene; False = sky) and frame.objects (pixels on things
standing on the terrain, e.g. boulders: mapped as standing obstacles whatever their class, never as ground;
with frame.object_depth they are placed at that measured depth instead of where their column meets the ground).

depth.source: depth_anything (monocular depth, corrected on the floor) | sam2 (terrain segmentation, best.pth:
obstacle pixels stand upright on the ground, free pixels lie on it; see projection.segmentation_depth) | sim.

Independent of where frames come from: anything with rgb, K, cam_pos, cam_rot (and optionally
a ground-truth depth) works, so the same pipeline runs on the simulator or the real rover.
"""

import os

import numpy as np
import yaml

from occupancy_grid import OccupancyGrid
from projection import (FloorCorrector, FloorPoseRefiner, JointFloorFit, backproject, edge_mask,
                        first_hit_filter, floor_depth, seg_cell_evidence, segmentation_depth)

HERE = os.path.dirname(os.path.abspath(__file__))


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def load_config(path=None, overrides=(), extra=()):
    """Load config.yaml, deep-merge any `extra` yaml files on top (e.g. sim/config_sim.yaml),
    then apply overrides like ["grid.resolution=0.05", "depth.source=sim"]."""
    with open(path or os.path.join(HERE, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    for e in extra or ():
        with open(e) as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or ():
        key, val = item.split("=", 1)
        *parents, leaf = key.split(".")
        d = cfg
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = yaml.safe_load(val)
    return cfg


class BEVPipeline:
    def __init__(self, cfg, depth_model=None):
        self.cfg = cfg
        dc = cfg["depth"]
        self.source = dc["source"]
        self.depth_model = depth_model
        if self.source == "depth_anything" and depth_model is None:
            from depth_estimator import DepthAnything
            self.depth_model = DepthAnything(dc["model_id"])
        self.seg_ids = None
        if self.source == "sam2":
            from sam2_seg import CLASS_NAMES, SAM2Segmenter
            sc = cfg.get("seg", {})
            if depth_model is None:
                ckpt = sc.get("checkpoint")
                if ckpt and not os.path.isabs(ckpt):
                    ckpt = os.path.join(HERE, ckpt)
                self.depth_model = SAM2Segmenter(ckpt, image_size=sc.get("image_size", 1024),
                                                 groups=sc.get("head_groups", 32), act=sc.get("head_act", "gelu"))
            ids = lambda names: [CLASS_NAMES.index(n) for n in names]  # noqa: E731
            self.seg_ids = (ids(sc.get("obstacle_classes", ["bigrock"])),
                            ids(sc.get("free_classes", ["soil", "bedrock", "sand"])), sc.get("min_conf", 0.5))
            self.ground_occ_ids = ids(sc.get("ground_occupied_classes", []) or [])
        self.corrector = FloorCorrector(dc["floor_correction"], dc["correction_ema"], dc["max_range"])
        self.grid = OccupancyGrid(cfg["grid"], cfg["classify"])
        pr = cfg.get("pose_refine", {})
        self.refiner = self.joint = None
        if pr.get("enabled", False) and self.source != "sam2":   # sam2: no depth to fit the floor on
            if self.source == "sim":    # exact depth: a plane fit gives the tilt directly
                self.refiner = FloorPoseRefiner(pr.get("max_tilt_deg", 8.0), pr.get("rel_tol", 0.04),
                                                pr.get("min_inlier_frac", 0.3), max_range=dc["max_range"])
            else:                       # Depth Anything: fit depth correction and tilt together
                self.joint = JointFloorFit(pr.get("max_tilt_deg", 8.0), dc["max_range"],
                                           ema=dc["correction_ema"],
                                           min_inlier_frac=pr.get("min_inlier_frac", 0.3),
                                           fit_pitch=pr.get("fit_pitch", False),
                                           gate_rel=dc.get("scale_gate_rel", 0.0),
                                           gate_window=dc.get("scale_gate_window", 7))
        self._floor_cache = None

    def reset(self):
        self.grid.reset()
        self.corrector.reset()
        if self.joint is not None:
            self.joint.reset()

    def _floor(self, frame):
        if getattr(frame, "floor_depth", None) is not None:   # uneven ground: depth of the known terrain per pixel
            return frame.floor_depth
        img = frame.rgb if getattr(frame, "rgb", None) is not None else frame.depth
        h, w = img.shape[:2]
        key = (h, w, *np.round(frame.cam_pos[2:], 3), *np.round(frame.cam_rot[2], 3))
        if self._floor_cache is None or self._floor_cache[0] != key:
            self._floor_cache = (key, floor_depth(frame.K, h, w, frame.cam_pos, frame.cam_rot))
        return self._floor_cache[1]

    def estimate_depth(self, frame):
        """Depth used for mapping (metres) + raw model output, for display/debug.

        With pose_refine on, the frame's camera rotation is first corrected (pitch/roll) from the
        floor seen in the raw depth, so the floor model below uses the camera's real tilt.
        """
        if self.source == "sam2":
            labels, probs = self.depth_model([frame.rgb])[0]
            obst_ids, free_ids, min_conf = self.seg_ids
            floor_d = self._floor(frame)
            self.last_labels = labels                       # as segmented (for display)
            conf = probs.max(0)
            objects = getattr(frame, "objects", None)      # pixels on things standing on the ground (not terrain)
            if objects is not None and obst_ids:
                labels = labels.copy()
                labels[objects] = obst_ids[0]               # -> a standing obstacle, never ground
                conf = np.where(objects, 1.0, conf)
            depth, _, _ = segmentation_depth(labels, conf, floor_d, obst_ids, free_ids, min_conf)
            od = getattr(frame, "object_depth", None)       # measured depth of those objects (if known): exact
            if objects is not None and od is not None:      # placement (a bulging rock hides where it meets the
                depth = np.where(objects & (od > 0), od, depth).astype(np.float32)   # ground)
            self.last_seg = (labels, conf, floor_d, depth)
            return depth, depth
        raw = frame.depth if self.source == "sim" else self.depth_model([frame.rgb])[0]
        if self.refiner is not None:
            frame.cam_rot, _ = self.refiner(raw, frame.K, frame.cam_rot)
        if self.source == "sim":
            return raw, raw
        if self.joint is not None:
            depth, frame.cam_rot = self.joint(raw, frame.K, frame.cam_pos, frame.cam_rot)
            return depth, raw
        return self.corrector(raw, self._floor(frame)), raw

    def fit_ok(self):
        """False when this frame's floor fit failed (its depth uses an older correction)."""
        if self.joint is not None:
            return bool(self.joint.last["ok"])
        if self.source in ("sim", "sam2") or self.corrector.mode == "none":
            return True
        return self.corrector.a is not None

    def process(self, frame, fuse=True):
        """Map one frame. fuse=False computes everything but leaves the map untouched
        (used to skip frames taken while the robot turns fast or bounces)."""
        dc, fc = self.cfg["depth"], self.cfg["filter"]
        depth, raw = self.estimate_depth(frame)
        # A frame whose floor fit failed would be mapped with a stale depth correction and an
        # unrefined tilt -> wrong heights. Skip it instead (option depth.skip_unfitted).
        fused = fuse and (self.fit_ok() or not dc.get("skip_unfitted", False))

        valid = np.isfinite(depth) & (depth > dc["min_range"]) & (depth < dc["max_range"])
        if self.source == "sam2":       # no flying pixels: the depth is built from the segmentation
            edges = np.zeros(depth.shape, bool)
        else:
            edges = edge_mask(depth, fc["edge_rel_grad"], fc["edge_dilate"])
        valid &= ~edges
        if getattr(frame, "mask", None) is not None:   # e.g. sim background (empty void)
            valid &= frame.mask

        pts, _, _ = backproject(depth, valid, frame.K, frame.cam_pos, frame.cam_rot, fc["stride"])
        if getattr(frame, "ground_z", None) is not None:      # uneven ground: heights above the known terrain
            pts = pts.copy()
            pts[:, 2] -= frame.ground_z(pts[:, 0], pts[:, 1])
        floor, obst = self.grid.classify(pts, frame.cam_pos[:2])
        fh = self.cfg.get("first_hit", {})
        if fh.get("enabled", True):
            obst = first_hit_filter(pts, obst, frame.cam_pos[:2], fh.get("bin_deg", 0.5),
                                    fh.get("quantile", 0.05), fh.get("min_rank", 2),
                                    fh.get("margin", 0.08), fh.get("margin_per_m", 0.04))
        if fused and self.source == "sam2":     # the map from the cells: each cell takes its pixel's class
            labels, conf, floor_d, rock_d = self.last_seg
            obst_ids, free_ids, min_conf = self.seg_ids
            G, R = self.grid, dc["max_range"]
            ix0, iy0, _ = G.world_to_cell(frame.cam_pos[0] - R, frame.cam_pos[1] - R)   # only cells in range
            ix1, iy1, _ = G.world_to_cell(frame.cam_pos[0] + R, frame.cam_pos[1] + R)
            win = (slice(max(int(iy0), 0), min(int(iy1) + 1, G.ny)), slice(max(int(ix0), 0), min(int(ix1) + 1, G.nx)))
            cx, cy = G.cell_x[win], G.cell_y[win]
            gz = getattr(frame, "ground_z", None)
            cz = gz(cx, cy) if gz is not None else np.zeros_like(cx)
            sc = self.cfg.get("seg", {})
            o, f = seg_cell_evidence(cx, cy, cz, labels, conf, floor_d, rock_d, frame.K, frame.cam_pos,
                                     frame.cam_rot, R, free_ids, self.ground_occ_ids, obst_ids, min_conf,
                                     getattr(frame, "mask", None), sc.get("rock_depth", 0.5),
                                     hidden_free=sc.get("terrain_hidden_free", False))
            occ_ev = np.zeros(G.L.shape, bool)
            free_ev = np.zeros(G.L.shape, bool)
            occ_ev[win], free_ev[win] = o, f
            oxy = getattr(frame, "object_xy", None)         # (N, 2) points on objects (map frame): their cells are
            if oxy is not None and len(oxy):                # occupied too (the ground under an overhang)
                ix, iy, ok = G.world_to_cell(oxy[:, 0], oxy[:, 1])
                occ_ev[iy[ok], ix[ok]] = True
            occ_ev, free_ev = G.fuse(occ_ev, free_ev)
        elif fused:
            occ_ev, free_ev = self.grid.update(pts, frame.cam_pos[:2], dc["max_range"], floor, obst)
        else:
            occ_ev = free_ev = np.zeros(self.grid.L.shape, bool)
        return {
            "depth": depth, "raw_depth": raw, "valid": valid, "edges": edges,
            "points": pts, "occ_ev": occ_ev, "free_ev": free_ev,
            "scale": self.joint.a if self.joint is not None else self.corrector.a,
            "offset": self.joint.b if self.joint is not None else self.corrector.b,
            "floor_inliers": (self.joint.last["inlier_frac"] if self.joint is not None
                              else self.corrector.last_inlier_frac),
            "fit_ok": self.fit_ok(), "fused": fused, "floor": floor, "obst": obst,
            "tilt_fix": (self.joint or self.refiner).last if (self.joint or self.refiner) else None,
            "labels": getattr(self, "last_labels", None) if self.source == "sam2" else None,
        }
