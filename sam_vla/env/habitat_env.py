from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import habitat_sim
import magnum as mn
import numpy as np
import quaternion
from habitat_sim.agent import AgentConfiguration

from sam_vla.core.goal_geometry import GoalPosition, terrain_patch_mesh
from sam_vla.core.lifecycle import ServiceRegistry
from sam_vla.core.types import Observation, Pose
from sam_vla.env.annotation_meshes import load_mesh_id_map, register_annotation_meshes
from sam_vla.env.mesh_terrain import NavmeshTerrain
from sam_vla.env.terrain import SIZE_X, SIZE_Y, SIZE_Z, HeightmapGrid, Terrain
from sam_vla.env.sim_utils import (
    make_sensor,
    register_glb_object,
    register_semantic_mesh,
    rgb_depth,
    save_obj,
    set_agent_pose,
    set_agent_pose_topdown,
    set_objects_hidden,
)
from sam_vla.env.pose_sweep import sample_in_place_sweep_yaws
from sam_vla.env.rock_generation import RockSpec, load_rock_field, register_rocks

RGB_HEIGHT = 480
RGB_WIDTH = 640
HFOV_DEG = 90.0
DEPTH_MAX_RANGE_M = 30.0

# Fixed bird's-eye viz camera (see enable_topdown_viz) -- human-viewing only,
# never fed to a policy/VLM/segmentation model, so it's not held to the same
# "match the rover sensor" resolution as RGB_HEIGHT/RGB_WIDTH above.
TOPDOWN_HEIGHT = 1080
TOPDOWN_WIDTH = 1080
TOPDOWN_HFOV_DEG = 90.0
# Extra clearance added around the flags' bounding box so no flag/marker
# sits flush against the frame edge.
TOPDOWN_MARGIN_M = 4.0

# assets/perseverance_mars_rover.glb is authored at real-world 1:1 meter
# scale (~2.7m wide x 2.3m tall x 4.5m long, matching the real Perseverance
# rover's proportions) -- 1.0 renders it true-to-scale against marsyard2022's
# already full-size course/flag spacing. Exposed as rover_marker_scale in
# case a future course/asset combination needs it dialed down.
ROVER_MARKER_GLB = (
    Path(__file__).resolve().parent.parent.parent / "assets" / "perseverance_mars_rover.glb"
)
ROVER_MARKER_SCALE = 1.0

SPAWN_CLEARANCE_M = 1.0
SPAWN_TERRAIN_RADIUS_M = 0.8


class MarsHabitatEnv:
    def __init__(
        self,
        scene_path: str,
        heightmap_path: Optional[str] = None,
        services: list = None,
        scene_dataset_config: Optional[str] = None,
        navmesh_path: Optional[str] = None,
        start_x: float = 0.0,
        start_z: float = 8.0,
        start_yaw: float = 0.0,
        randomise_spawn: bool = False,
        spawn_clearance: float = SPAWN_CLEARANCE_M,
        spawn_terrain_radius: float = SPAWN_TERRAIN_RADIUS_M,
        with_semantic: bool = False,
        gen_home_base: bool = False,
        rock_field_path: Optional[str] = None,
        annotations_dir: Optional[str] = None,
        annotation_categories: Optional[Sequence[str]] = None,
        flag_seed: Optional[int] = None,
        num_flags: int = 6,
        flag_min_spacing: float = 1.5,
        flag_boundary_margin: float = 2.0,
        flag_spawn_clearance: float = 2.0,
        spawn_flags: bool = True,
        enable_topdown_viz: bool = False,
        topdown_resolution: tuple = (TOPDOWN_HEIGHT, TOPDOWN_WIDTH),
        topdown_hfov_deg: float = TOPDOWN_HFOV_DEG,
        topdown_margin_m: float = TOPDOWN_MARGIN_M,
        rover_marker_scale: float = ROVER_MARKER_SCALE,
        scenery: Optional[list] = None,
        terrain_glb: Optional[str] = None,
    ):
        # scenery: list of dicts {glb, x, z, yaw, scale, half_height} --
        # static render-only decoration (see sam_vla/env/scenery.py).
        self._scenery = list(scenery or [])
        # Take terrain heights from this GLB's own ground mesh (see
        # sam_vla/env/glb_terrain.py) instead of a heightmap/navmesh.
        self._terrain_glb = terrain_glb
        self.scenery_objects: list = []
        self._scene_path = Path(scene_path)
        self._heightmap_path = Path(heightmap_path) if heightmap_path else None
        self._scene_dataset_config = scene_dataset_config
        self._navmesh_path = navmesh_path
        self._registry = ServiceRegistry()
        for service in services or []:
            self._registry.register(service)
        self._sim = None
        self._agent = None
        self._terrain = None
        self._start_x = start_x
        self._start_z = start_z
        self._start_yaw = start_yaw
        self._randomise_spawn = randomise_spawn
        self._spawn_clearance = spawn_clearance
        self._spawn_terrain_radius = spawn_terrain_radius
        self._with_semantic = with_semantic
        self._gen_home_base = gen_home_base
        self._rock_field_path = Path(rock_field_path) if rock_field_path else None
        self._annotations_dir = Path(annotations_dir) if annotations_dir else None
        self._annotation_categories = annotation_categories
        # None disables flag placement entirely (default) -- generation is live/seeded
        # rather than a cached manifest (see sam_vla.env.flag_placement), so there's
        # nothing to opt into by path the way rock_field_path works.
        self._flag_seed = flag_seed
        self._num_flags = int(num_flags)
        self._flag_min_spacing = float(flag_min_spacing)
        self._flag_boundary_margin = float(flag_boundary_margin)
        self._flag_spawn_clearance = float(flag_spawn_clearance)
        # False skips flag placement entirely (self.flags stays []) -- for
        # callers that don't want the decorative diamond cluttering a scene,
        # e.g. an offline rollout replay framed around its own recorded
        # trajectory/goal/obstacle instead.
        self._spawn_flags = bool(spawn_flags)
        self._enable_topdown_viz = bool(enable_topdown_viz)
        self._topdown_resolution = (
            int(topdown_resolution[0]),
            int(topdown_resolution[1]),
        )
        self._topdown_hfov_deg = float(topdown_hfov_deg)
        self._topdown_margin_m = float(topdown_margin_m)
        self._topdown_agent = None
        self._topdown_center: Optional[Tuple[float, float]] = None
        self._topdown_cam_height: Optional[float] = None
        self._rover_marker_scale = float(rover_marker_scale)
        # A render-only Perseverance model tracking the driving agent's live
        # pose, shown only in the topdown camera's frame (see
        # get_topdown_rgb's hide/show toggle) -- never the driving agent's
        # own camera, which sits inside the model's own body at that same
        # position/scale and would otherwise render from its interior.
        self._rover_marker_obj = None
        self._rover_marker_onstage = None
        self._rover_marker_half_height = 0.0
        self.rocks: List[RockSpec] = []
        self.flags: list = []
        self.home_base = None
        self.annotation_mesh_id_map: dict = {}
        self._annotation_objects: dict = {}
        self._annotation_onstage: dict = {}

    def __enter__(self) -> "MarsHabitatEnv":
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(self._scene_path.expanduser().resolve())
        sim_cfg.enable_physics = False
        if self._scene_dataset_config is not None:
            sim_cfg.scene_dataset_config_file = str(
                Path(self._scene_dataset_config).expanduser().resolve()
            )

        rgb_spec = make_sensor(
            "rgb", habitat_sim.SensorType.COLOR, RGB_HEIGHT, RGB_WIDTH, HFOV_DEG
        )
        depth_spec = make_sensor(
            "depth", habitat_sim.SensorType.DEPTH, RGB_HEIGHT, RGB_WIDTH, HFOV_DEG
        )
        # make_sensor doesn't set a far clip; cap depth range here to a sane sensor spec.
        depth_spec.far = DEPTH_MAX_RANGE_M

        sensor_specs = [rgb_spec, depth_spec]
        if self._with_semantic:
            sensor_specs.append(
                make_sensor(
                    "semantic",
                    habitat_sim.SensorType.SEMANTIC,
                    RGB_HEIGHT,
                    RGB_WIDTH,
                    HFOV_DEG,
                )
            )

        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        agent_cfgs = [agent_cfg]
        if self._enable_topdown_viz:
            topdown_h, topdown_w = self._topdown_resolution
            topdown_spec = make_sensor(
                "topdown",
                habitat_sim.SensorType.COLOR,
                topdown_h,
                topdown_w,
                self._topdown_hfov_deg,
            )
            topdown_agent_cfg = AgentConfiguration()
            topdown_agent_cfg.sensor_specifications = [topdown_spec]
            agent_cfgs.append(topdown_agent_cfg)

        self._sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, agent_cfgs)
        )
        self._agent = self._sim.initialize_agent(0)
        if self._enable_topdown_viz:
            self._topdown_agent = self._sim.initialize_agent(1)

        if self._terrain_glb is not None:
            from sam_vla.env.glb_terrain import GlbTerrain

            self._terrain = GlbTerrain(self._terrain_glb)
        elif self._heightmap_path is not None:
            heightmap_grid = HeightmapGrid(
                self._heightmap_path.expanduser().resolve(),
                size_x=SIZE_X,
                size_z=SIZE_Z,
                size_y=SIZE_Y,
                flip_x=False,
                flip_z=True,
                swap_xz=False,
            )
            self._terrain = Terrain(
                heightmap_grid, flip_x=False, flip_z=False, swap_xz=False
            )
        else:
            if self._navmesh_path is not None and not self._sim.pathfinder.is_loaded:
                self._sim.pathfinder.load_nav_mesh(
                    str(Path(self._navmesh_path).expanduser().resolve())
                )
            if not self._sim.pathfinder.is_loaded:
                raise ValueError(
                    "no heightmap_path given and no navmesh loaded (neither an "
                    "explicit navmesh_path nor the scene's own sibling .navmesh "
                    "auto-loaded) -- MarsHabitatEnv needs one height source or "
                    "the other"
                )
            seed_point = self._sim.pathfinder.get_random_navigable_point()
            self._terrain = NavmeshTerrain(
                self._sim.pathfinder, seed_y=float(seed_point[1])
            )

        if self._randomise_spawn:
            x = random.uniform(-SIZE_X / 2.0, SIZE_X / 2.0)
            z = random.uniform(-SIZE_Z / 2.0, SIZE_Z / 2.0)
            yaw = random.uniform(0.0, 2.0 * 3.141592653589793)
        else:
            x, z, yaw = self._start_x, self._start_z, self._start_yaw

        y = self.get_height_at_xz(x, z)
        set_agent_pose(self._agent, x, y, z, yaw)

        if self._gen_home_base:
            from sam_vla.env.home_base import register_home_base

            self.home_base = register_home_base(self._sim, self._terrain, x, z, yaw)

        if self._scenery:
            from sam_vla.env.scenery import register_scenery

            for item in self._scenery:
                self.scenery_objects.append(
                    register_scenery(self._sim, self._terrain, **item)
                )

        if self._rock_field_path is not None:
            self.rocks, _rock_config = load_rock_field(self._rock_field_path)
            register_rocks(self._sim, self.rocks)

        if self._spawn_flags:
            if self._flag_seed is not None:
                from sam_vla.env.flag_placement import (
                    FlagFieldConfig,
                    generate_flag_field,
                    register_flags,
                )

                # Always keep flags clear of the rover's own spawn point, on
                # top of any caller-supplied exclude zones.
                flag_config = FlagFieldConfig(
                    seed=self._flag_seed,
                    num_flags=self._num_flags,
                    min_spacing=self._flag_min_spacing,
                    boundary_margin=self._flag_boundary_margin,
                    exclude_zones=[(x, z, self._flag_spawn_clearance)],
                )
                self.flags = generate_flag_field(
                    flag_config, self._terrain, face_target=(x, z)
                )
                register_flags(self._sim, self.flags)
            else:
                # No seed given: fall back to a fixed four-flag diamond around
                # spawn (see flag_placement.generate_default_flag_field) instead
                # of leaving the yard empty -- a "go left find a flag" mission
                # prompt always has something to find.
                from sam_vla.env.flag_placement import (
                    generate_default_flag_field,
                    register_flags,
                )

                self.flags = generate_default_flag_field(self._terrain, x, z, yaw)
                register_flags(self._sim, self.flags)

        if self._topdown_agent is not None:
            self._place_topdown_camera(fallback_x=x, fallback_z=z)
            self._register_rover_marker(x, z, yaw)

        if self._annotations_dir is not None:
            self.annotation_mesh_id_map = load_mesh_id_map(self._annotations_dir)
            self._annotation_objects = register_annotation_meshes(
                self._sim, str(self._annotations_dir), self._annotation_categories
            )
            self._annotation_onstage = {
                mesh_id: obj.translation
                for mesh_id, obj in self._annotation_objects.items()
            }
            # Hidden by default (rather than onstage/visible, as
            # register_annotation_meshes leaves them) -- callers outside the
            # run_segmentation_sweep capture path (e.g. nav/rover_controller's
            # live loop, which only ever calls get_observation, never
            # get_full_observation's hide/show dance) must never have these
            # meshes leak into a frame unless they explicitly ask for one via
            # get_mesh_overlay_rgb.
            set_objects_hidden(
                self._annotation_objects, self._annotation_onstage, hidden=True
            )

        self._registry.start_all()
        return self

    def _place_topdown_camera(
        self,
        fallback_x: float,
        fallback_z: float,
        extra_points: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        """Point the fixed viz-only top-down camera (see enable_topdown_viz)
        straight down at a height/position chosen so its frame's ground
        footprint contains every flag's (x, z), plus any caller-supplied
        `extra_points` (e.g. an offline rollout's full trajectory/goal/
        obstacle positions -- see refit_topdown_camera), plus
        topdown_margin_m of clearance. Falls back to a small box around the
        rover's spawn point if neither flags nor extra_points are given."""
        xs = [f.x for f in self.flags]
        zs = [f.z for f in self.flags]
        if extra_points:
            for px, pz in extra_points:
                xs.append(float(px))
                zs.append(float(pz))
        if xs:
            min_x, max_x = min(xs), max(xs)
            min_z, max_z = min(zs), max(zs)
        else:
            min_x = max_x = fallback_x
            min_z = max_z = fallback_z

        margin = self._topdown_margin_m
        half_width = max((max_x - min_x) / 2.0 + margin, margin)
        half_depth = max((max_z - min_z) / 2.0 + margin, margin)
        cx = (min_x + max_x) / 2.0
        cz = (min_z + max_z) / 2.0

        topdown_h, topdown_w = self._topdown_resolution
        hfov = math.radians(self._topdown_hfov_deg)
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * topdown_h / topdown_w)

        height_for_width = half_width / math.tan(hfov / 2.0)
        height_for_depth = half_depth / math.tan(vfov / 2.0)
        cam_height = max(height_for_width, height_for_depth)

        ground_y = self._terrain(cx, cz)
        set_agent_pose_topdown(self._topdown_agent, cx, ground_y + cam_height, cz)
        # Cached for project_world_to_topdown_pixel -- the *actual* ground
        # footprint at this cam_height, not half_width/half_depth themselves
        # (whichever dimension isn't the constraining one ends up with a
        # larger visible footprint than its own requested half-extent, since
        # both dimensions share one cam_height).
        self._topdown_center = (cx, cz)
        self._topdown_cam_height = cam_height

    def refit_topdown_camera(
        self, extra_xz: Optional[Sequence[Tuple[float, float]]] = None
    ) -> None:
        """Recompute the fixed top-down camera's framing to also cover
        `extra_xz` (world (x, z) points) -- for offline rollout replay,
        where the interesting region (a saved episode's full trajectory,
        goal, obstacles) can extend well beyond wherever the default flag
        diamond landed, or beyond nothing at all if spawn_flags=False.
        No-op if enable_topdown_viz=False."""
        if self._topdown_agent is None:
            return
        self._place_topdown_camera(
            self._start_x, self._start_z, extra_points=extra_xz
        )

    def project_world_to_topdown_pixel(
        self, x: float, z: float
    ) -> Optional[Tuple[int, int]]:
        """World (x, z) -> pixel (col, row) in the top-down camera's frame,
        for drawing 2D overlays (goal/obstacle markers, trajectory trails)
        on top of get_topdown_rgb's render -- mirrors goal_geometry.
        project_world_to_pixel's role for the driving agent's own camera.
        Not clamped to the frame bounds; callers should check against
        topdown_resolution before drawing. None if enable_topdown_viz=False."""
        if self._topdown_agent is None or self._topdown_cam_height is None:
            return None
        cx, cz = self._topdown_center
        h = self._topdown_cam_height
        topdown_h, topdown_w = self._topdown_resolution
        hfov = math.radians(self._topdown_hfov_deg)
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * topdown_h / topdown_w)
        half_w = h * math.tan(hfov / 2.0)
        half_d = h * math.tan(vfov / 2.0)
        px = (topdown_w / 2.0) + (float(x) - cx) / half_w * (topdown_w / 2.0)
        py = (topdown_h / 2.0) + (float(z) - cz) / half_d * (topdown_h / 2.0)
        return int(round(px)), int(round(py))

    def _register_rover_marker(self, x: float, z: float, yaw: float) -> None:
        """Register the render-only Perseverance model at the rover's spawn
        pose, resting state hidden (see sim_utils.set_objects_hidden -- same
        drop-far-away trick the annotation hull meshes use, since habitat-sim
        has no per-sensor visibility flag): _sync_rover_marker +
        get_topdown_rgb are the only things that ever show it, and only for
        the duration of the topdown camera's own render call."""
        ground_y = self._terrain(x, z)
        obj = register_glb_object(
            self._sim,
            str(ROVER_MARKER_GLB),
            (x, ground_y, z),
            yaw=yaw,
            scale=self._rover_marker_scale,
        )
        # Model origin is at its own vertical center, not its wheelbase --
        # lift by half its (already-scaled) height so it rests on the
        # terrain instead of sinking halfway into it.
        self._rover_marker_half_height = float(obj.aabb.size().y) / 2.0
        obj.translation = obj.translation + mn.Vector3(
            0.0, self._rover_marker_half_height, 0.0
        )
        self._rover_marker_obj = obj
        self._rover_marker_onstage = obj.translation
        set_objects_hidden(
            {0: obj}, {0: self._rover_marker_onstage}, hidden=True
        )

    def _sync_rover_marker(self) -> None:
        """Move the (still-hidden) rover marker to the driving agent's
        current live pose -- called right before get_topdown_rgb shows it,
        so the marker always reflects this tick's position/heading rather
        than wherever the rover was at spawn."""
        state = self._agent.get_state()
        x, _y, z = (float(v) for v in state.position)
        yaw = float(quaternion.as_rotation_vector(state.rotation)[1])
        ground_y = self._terrain(x, z)
        pos = mn.Vector3(x, ground_y + self._rover_marker_half_height, z)
        self._rover_marker_obj.translation = pos
        self._rover_marker_obj.rotation = mn.Quaternion.rotation(
            mn.Rad(yaw), mn.Vector3.y_axis()
        )
        self._rover_marker_onstage = pos

    def get_topdown_rgb(self) -> Optional[np.ndarray]:
        """RGB frame from the fixed bird's-eye viz camera (see
        enable_topdown_viz), framing the flag field from directly above,
        with the rover marker (see _register_rover_marker) shown at its
        current live pose for exactly this one render. Human-viewing/
        recording only -- never fed to a policy, VLM, or segmentation
        model. Returns None if enable_topdown_viz=False."""
        if self._topdown_agent is None:
            return None
        if self._rover_marker_obj is not None:
            self._sync_rover_marker()
            set_objects_hidden(
                {0: self._rover_marker_obj},
                {0: self._rover_marker_onstage},
                hidden=False,
            )
        obs = self._sim.get_sensor_observations(agent_ids=1)
        rgb = np.asarray(obs["topdown"])
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        rgb = rgb.astype(np.uint8)
        if self._rover_marker_obj is not None:
            set_objects_hidden(
                {0: self._rover_marker_obj},
                {0: self._rover_marker_onstage},
                hidden=True,
            )
        return rgb

    def get_height_at_xz(self, x: float, z: float) -> float:
        """Terrain height at (x, z), plus rover clearance, sampled from the heightmap."""
        return (
            self._terrain.local_height_max(x, z, self._spawn_terrain_radius)
            + self._spawn_clearance
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._registry.stop_all()
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        return False

    def get_observation(self, frame_idx: int) -> Observation:
        obs = self._sim.get_sensor_observations()
        rgb, depth = rgb_depth(obs)
        state = self._agent.get_state()
        x, _y, z = (float(v) for v in state.position)
        yaw = float(quaternion.as_rotation_vector(state.rotation)[1])
        pose = Pose(x=x, y=float(state.position[1]), z=z, yaw=yaw)
        return Observation(rgb=rgb, depth=depth, pose=pose, frame_idx=frame_idx)

    def get_mesh_overlay_rgb(self) -> Optional[np.ndarray]:
        """RGB captured with the annotation hull meshes visible in the color
        pass -- i.e. the inverse of get_full_observation's Pass A, and the
        same composited appearance the mesh_tight_bound2-trained LoRA
        segmentation checkpoint (sam_lora_runs/exp10) was trained against.
        For feeding a segmentation model only: never returned by
        get_observation/get_full_observation, so it never reaches the GUI,
        the VLM, or anything else that treats a frame as "what the rover
        sees". Returns None if no annotations_dir was configured. Meshes are
        restored to hidden before returning, matching this env's resting
        state (see __enter__)."""
        if not self._annotation_objects:
            return None
        set_objects_hidden(
            self._annotation_objects, self._annotation_onstage, hidden=False
        )
        obs = self._sim.get_sensor_observations()
        rgb, _depth = rgb_depth(obs)
        set_objects_hidden(
            self._annotation_objects, self._annotation_onstage, hidden=True
        )
        return rgb

    def get_full_observation(self, frame_idx: int) -> Observation:
        """Returns rgb+depth+pose (as get_observation does) AND the semantic
        id buffer (as get_semantic_frame does). `.semantic` is None if
        with_semantic=False (mirrors get_semantic_frame's precondition).

        If annotation meshes are loaded, this does TWO render passes instead
        of one: Pass A (rgb/depth) is captured with the annotation meshes
        moved out of camera view via sim_utils.set_objects_hidden, so they
        can never bake into the training RGB (see next.md -- the mesh sits
        a few cm above terrain for the semantic pass and was, before this
        split, also rendering into RGB as a visible object, producing a
        deterministic occlusion artifact at every positive location that a
        detector could trivially key on instead of real rock appearance).
        Pass B (semantic only) is then captured with the meshes back
        onstage. With no annotation meshes loaded, this is a single render
        call exactly as before -- get_observation() and get_semantic_frame()
        each trigger their own render call, so a caller needing both should
        still prefer this over calling them separately."""
        state = self._agent.get_state()
        x, _y, z = (float(v) for v in state.position)
        yaw = float(quaternion.as_rotation_vector(state.rotation)[1])
        pose = Pose(x=x, y=float(state.position[1]), z=z, yaw=yaw)

        if self._annotation_objects:
            set_objects_hidden(
                self._annotation_objects, self._annotation_onstage, hidden=True
            )
            obs = self._sim.get_sensor_observations()
            rgb, depth = rgb_depth(obs)

            set_objects_hidden(
                self._annotation_objects, self._annotation_onstage, hidden=False
            )
            obs2 = self._sim.get_sensor_observations()
            semantic = np.asarray(obs2["semantic"]) if self._with_semantic else None
        else:
            obs = self._sim.get_sensor_observations()
            rgb, depth = rgb_depth(obs)
            semantic = np.asarray(obs["semantic"]) if self._with_semantic else None

        return Observation(
            rgb=rgb, depth=depth, pose=pose, frame_idx=frame_idx, semantic=semantic
        )

    def verify_annotation_isolation(self, frame_idx: int = -1) -> None:
        """Standing regression check (next.md Step 5): assert
        get_full_observation's actual Pass A RGB (the production path)
        matches an explicit "annotation meshes absent" control render, at
        the current agent pose. This is NOT the same as asserting
        onstage-RGB == hidden-RGB -- those are *expected* to differ
        whenever a mesh is in frame, since a real object was moved out of
        view; that difference is the artifact next.md describes; it's fine.
        What must never happen is get_full_observation itself forgetting to
        hide the meshes before capturing Pass A, i.e. Pass A silently
        matching the "meshes onstage" render instead of the "meshes absent"
        one. Raises AssertionError if that ever happens. No-op if no
        annotation meshes are loaded."""
        if not self._annotation_objects:
            return

        obs = self.get_full_observation(frame_idx)  # production path

        set_objects_hidden(
            self._annotation_objects, self._annotation_onstage, hidden=True
        )
        obs_absent = self._sim.get_sensor_observations()
        rgb_absent, _ = rgb_depth(obs_absent)
        set_objects_hidden(
            self._annotation_objects, self._annotation_onstage, hidden=False
        )  # restore

        if not np.array_equal(obs.rgb, rgb_absent):
            diff_pixels = int(np.any(obs.rgb != rgb_absent, axis=-1).sum())
            raise AssertionError(
                f"annotation mesh isolation broken: get_full_observation's RGB differs by "
                f"{diff_pixels} pixels from a render with annotation meshes explicitly absent -- "
                "the annotation meshes are leaking into the training RGB again (see next.md)."
            )

    def get_semantic_frame(self) -> np.ndarray:
        """Raw per-pixel semantic-id image (H, W) from the semantic sensor:
        each pixel holds the semantic_id of whatever registered mask mesh is
        rendered there (see register_object_mask / goal_geometry.MESH_GOAL_ID
        / MESH_OBST_ID), 0 elsewhere. Requires with_semantic=True."""
        obs = self._sim.get_sensor_observations()
        return np.asarray(obs["semantic"])

    def sweep_in_place(
        self, num_yaws: int = 8, frame_idx_start: int = 0
    ) -> List[Observation]:
        """Rotate the agent through one full 360deg turn at its current
        (x, z) -- no translation -- capturing one get_full_observation()
        (rgb+depth+pose, plus semantic if with_semantic=True) per heading.
        For a script that wants to look around from a fixed spot rather than
        drive anywhere (e.g. surveying what's around the rover at a point),
        not for driving/rollout loops. Restores the original heading before
        returning. Headings come from sample_in_place_sweep_yaws, starting at
        (and including) the current yaw."""
        state = self._agent.get_state()
        x, _y, z = (float(v) for v in state.position)
        original_yaw = float(quaternion.as_rotation_vector(state.rotation)[1])

        observations = []
        for i, yaw in enumerate(
            sample_in_place_sweep_yaws(num_yaws, start_yaw=original_yaw)
        ):
            self.step(Pose(x=x, y=0.0, z=z, yaw=yaw))
            observations.append(self.get_full_observation(frame_idx=frame_idx_start + i))

        self.step(Pose(x=x, y=0.0, z=z, yaw=original_yaw))
        return observations

    def step(self, pose: Pose) -> None:
        # Match spawn's ground offset (local-max + clearance), not a raw single-point
        # sample, or the agent snaps to bare terrain height every step and clips into
        # the surface.
        y = self.get_height_at_xz(pose.x, pose.z)
        set_agent_pose(self._agent, pose.x, y, pose.z, pose.yaw)

    def register_object_mask(
        self,
        world_pos: GoalPosition,
        semantic_id: int,
        radius: float,
        out_dir: str,
        name: str,
    ):
        """Register a small terrain-following patch mesh at world_pos as a
        render-only, non-collidable object carrying `semantic_id`, so the
        semantic sensor renders a goal/obstacle mask around that point.
        Seeded from an already-backprojected world point
        (goal_geometry.bbox_to_world) rather than a raw pixel + depth patch;
        the patch itself is resampled from the terrain heightmap (self._terrain)
        so it hugs the ground instead of sitting on one flat plane. Requires
        with_semantic=True.
        """
        verts, faces = terrain_patch_mesh(world_pos, radius, self._terrain)
        mesh_dir = Path(out_dir) / "masks"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = str(mesh_dir / f"{name}.obj")
        save_obj(mesh_path, verts, faces)
        return register_semantic_mesh(self._sim, mesh_path, semantic_id)

    def remove_object_mask(self, obj) -> None:
        """Undo a prior register_object_mask outright (not just untag it via
        semantic_id=0) -- callers that repeatedly resolve/rerun within one
        run must actually delete the old mesh, or it keeps sitting in the
        scene as dead render geometry forever."""
        self._sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent.parent.parent
    scene = HERE / "assets" / "marsyard2022.glb"
    heightmap = HERE / "marsyard2022_terrain_hm_1025.tif"

    with MarsHabitatEnv(str(scene), str(heightmap), services=[]) as env:
        obs = env.get_observation(0)
        print(f"rgb shape={obs.rgb.shape} dtype={obs.rgb.dtype}")
        print(f"pose={obs.pose}")
