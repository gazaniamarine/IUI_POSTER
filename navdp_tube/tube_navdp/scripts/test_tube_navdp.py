#!/usr/bin/env python3
"""CPU tests for tube-guided NavDP (navdp/tube): the grid tubes (no sides, no circles), the one-pass score-guided
sampler with a stand-in denoiser, the agent wrapper with a stand-in policy, the client, the controller and the driver
with a stand-in planner (no NavDP weights, no server)."""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from navdp.official_client import NavDPResult
from navdp.tube.stt import (Bev, TubeConfig, clusters, inside, min_width_at, start_box, sweep, tubes_for,
                            waypoint_times)

try:
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except ImportError:
    torch = None

ROCK = (3.0, 0.0, 1.0)  # forward, left, radius (the rover's size included)


def disc_bev(discs, resolution=0.05, half_m=6.5):
    """A body-frame grid with round blobs of occupied cells (forward, left, radius): any shape of map would do."""
    n = int(round(2 * half_m / resolution))
    centres = -half_m + (np.arange(n) + 0.5) * resolution
    forward, left = np.meshgrid(centres, centres, indexing="ij")
    occupied = np.zeros((n, n), bool)
    for of, ol, r in discs:
        reach = r + resolution / np.sqrt(2)   # every cell the disc touches
        occupied |= (forward - of) ** 2 + (left - ol) ** 2 <= reach * reach
    return Bev(occupied, resolution, (-half_m, -half_m))


def rock_bev():
    return disc_bev([ROCK])


def detour(side, amp=1.8, offset=0.0, horizon=24):
    """A path 5.3 m ahead with a sideways bump round the rock, made drivable (0.25 m per step)."""
    forward = 0.22 * (np.arange(horizon) + 1)
    left = side * (amp * np.exp(-((forward - ROCK[0]) / 1.2) ** 2) + offset)
    return sweep(np.stack([forward, left], axis=1), 0.25)


def scheduler():
    return DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", clip_sample=True,
                         prediction_type="epsilon")


def straight_through(n=16, horizon=24, length=5.5, lateral=0.3):
    """Stand-in denoiser: every sample's clean moves head straight to (length, +-lateral), through the rock."""
    ends = np.where(np.arange(n) % 2 == 0, lateral, -lateral)
    target = np.zeros((n, horizon, 3))
    target[..., 0] = length / horizon * 4
    target[..., 1] = (ends / horizon * 4)[:, None]
    target = torch.tensor(target, dtype=torch.float32)
    sched = scheduler()

    def predict(x, k):
        ab = sched.alphas_cumprod[int(k)]
        return (x - ab.sqrt() * target) / (1 - ab).sqrt()
    return predict, sched


def boxes_clear(tube, bev, n=500):
    lo, hi = tube.bounds(np.linspace(0, 1, n + 1))
    return not any(bev.box_blocked(a, b) for a, b in zip(lo, hi))


class Grid(unittest.TestCase):
    def test_encode_round_trip(self):
        bev = rock_bev()
        back = Bev.decode(json.loads(json.dumps(bev.encode())))
        np.testing.assert_array_equal(back.occupied, bev.occupied)
        self.assertEqual((back.resolution, back.origin), (bev.resolution, bev.origin))

    def test_cells(self):
        bev = rock_bev()
        self.assertTrue(bev.box_blocked([3.0, 0.99], [3.0, 0.99]))    # just inside the rim
        self.assertFalse(bev.box_blocked([3.0, 1.1], [3.0, 1.1]))     # outside
        self.assertTrue(bev.box_blocked([7.0, 0.0], [7.0, 0.0]))      # outside the grid counts as occupied

    def test_clusters_do_not_reach_across_an_obstacle(self):
        occupied = np.zeros((260, 260), bool)
        occupied[150, 128:132] = True                                  # a thin bar at forward ~1.0, left ~0
        bev = Bev(occupied, 0.05, (-6.5, -6.5))
        points = np.array([[1.02, -0.14], [1.02, 0.14], [1.02, 0.24]])
        groups = sorted(sorted(c.tolist()) for c in clusters(points, bev, 0.3))
        self.assertEqual(groups, [[0], [1, 2]])                        # close, but the bar is between 0 and 1


class TubeProgram(unittest.TestCase):
    def test_paths_round_a_rock_get_valid_clear_tubes(self):
        cfg, bev = TubeConfig(), rock_bev()
        paths = np.stack([detour(1), detour(1, offset=0.1), detour(-1), detour(-1, offset=0.1)])
        tubes, group, used = tubes_for(paths, bev, cfg, push=False)
        np.testing.assert_allclose(used, paths)
        self.assertNotEqual(group[0], group[2])                        # left and right never share a tube
        for g, tube in enumerate(tubes):
            self.assertTrue(tube.solved, g)
            self.assertTrue(tube.theorem, (tube.slack, tube.lipschitz, tube.gap))
            self.assertTrue(tube.feasible, tube.eta)
            self.assertTrue(boxes_clear(tube, bev))                    # every s, not only the sampled ones
            self.assertTrue(inside(paths[group == g], tube).all())
            dense = np.linspace(0, 1, 1001)[1:]
            lo, hi = tube.bounds(dense)
            self.assertTrue((hi - lo >= min_width_at(dense, cfg)[:, None] - cfg.eta_tol).all())
            wlo, whi = tube.bounds(np.concatenate([[0.0], waypoint_times(24)]))
            self.assertLessEqual(np.abs(np.diff(wlo, axis=0)).max(), cfg.rate * cfg.step_m + 1e-6)
            self.assertLessEqual(np.abs(np.diff(whi, axis=0)).max(), cfg.rate * cfg.step_m + 1e-6)
            s_lo, s_hi = start_box(cfg)
            np.testing.assert_allclose(tube.bounds([0.0])[0][0], s_lo, atol=1e-6)
            np.testing.assert_allclose(tube.bounds([0.0])[1][0], s_hi, atol=1e-6)

    def test_any_shape_works_a_wall_with_a_gap(self):
        res, half = 0.05, 6.5
        n = int(2 * half / res)
        centres = -half + (np.arange(n) + 0.5) * res
        forward, left = np.meshgrid(centres, centres, indexing="ij")
        wall = (np.abs(forward - 2.5) < 0.15) & (np.abs(left) > 0.6)   # a wall across the way, 1.2 m gap in it
        bev = Bev(wall, res, (-half, -half))
        path = sweep(np.stack([0.22 * np.arange(1, 25), np.zeros(24)], axis=1), 0.25)
        tubes, group, _ = tubes_for(path[None], bev, TubeConfig(), push=False)
        self.assertTrue(tubes[0].feasible, (tubes[0].eta, tubes[0].slack, tubes[0].lipschitz))
        self.assertTrue(boxes_clear(tubes[0], bev))

    def test_path_through_the_rock_is_pushed_clear(self):
        bev = rock_bev()
        straight = sweep(np.stack([0.22 * np.arange(1, 25), np.zeros(24)], axis=1), 0.25)
        tubes, group, used = tubes_for(straight[None], bev, TubeConfig(), push=True)
        self.assertFalse(any(bev.box_blocked(p, p) for p in used[0]))
        self.assertLessEqual(np.abs(np.diff(used[0], axis=0)).max(), 0.25 + 1e-9)
        # The same path, not pushed (as the certificate does with finished paths): no tube.
        tubes, group, _ = tubes_for(straight[None], bev, TubeConfig(), push=False)
        self.assertFalse(tubes[group[0]].feasible)

    def test_rover_inside_the_grown_obstacles_gets_no_tube(self):
        bev = disc_bev([(0.1, 0.0, 0.3)])
        tubes, group, _ = tubes_for(detour(1)[None], bev, TubeConfig(), push=False)
        self.assertEqual(len(tubes), 1)
        self.assertFalse(tubes[0].feasible)


@unittest.skipIf(torch is None, "needs torch and diffusers")
class GuidedSampler(unittest.TestCase):
    def test_sweep_lands_inside_rate_limited_boxes(self):
        from navdp.tube.guided_sampler import sweep_into_boxes
        g = torch.Generator().manual_seed(1)
        centre = torch.cumsum(torch.rand(8, 24, 2, generator=g) * 0.4 - 0.2, dim=1)
        lo, hi = centre - 0.2, centre + 0.2
        p = sweep_into_boxes(torch.randn(8, 24, 2, generator=g) * 3, lo, hi)
        self.assertTrue(((p >= lo - 1e-6) & (p <= hi + 1e-6)).all())
        steps = torch.diff(p, dim=1, prepend=torch.zeros_like(p[:, :1])).abs()
        self.assertLessEqual(float(steps.max()), 0.25 + 1e-6)

    def test_modes(self):
        from navdp.tube.guided_sampler import GuideConfig, choose, guided_denoise, positions
        predict, sched = straight_through()
        bev, results = rock_bev(), {}
        for mode in ("navdp", "obstacle", "tube"):
            out = guided_denoise(predict, sched, (16, 24, 3), "cpu", bev, GuideConfig(mode=mode),
                                 torch.Generator().manual_seed(0))
            p = positions(out["moves"]).numpy()
            results[mode] = (out, np.linalg.norm(p - np.array(ROCK[:2]), axis=2).min())
        self.assertLess(results["navdp"][1], 0.5)                 # the stand-in drives through the rock
        self.assertEqual(int(results["navdp"][0]["certified"].sum()), 0)
        tube, clearance = results["tube"]
        self.assertGreater(int(tube["certified"].sum()), 0)
        self.assertGreaterEqual(clearance, ROCK[2] - 0.05)        # every sample ends clear (one cell of slack)
        self.assertEqual(tube["split_paths"].shape, (16, 24, 2))
        self.assertGreater(results["obstacle"][1], ROCK[2] - 0.1)
        # Recording: the clean-path estimate at all 10 steps; at the last step it is the final path.
        rec = guided_denoise(predict, sched, (16, 24, 3), "cpu", bev, GuideConfig(mode="navdp"),
                             torch.Generator().manual_seed(0), record=True)
        self.assertEqual(rec["x0_steps"].shape, (10, 16, 24, 2))
        np.testing.assert_allclose(rec["x0_steps"][-1], positions(rec["moves"]).numpy(), atol=1e-4)
        # Same noise and no obstacle: the guided loop is exactly NavDP's loop.
        plain = guided_denoise(predict, sched, (16, 24, 3), "cpu", None, GuideConfig(mode="tube"),
                               torch.Generator().manual_seed(0))
        ref = guided_denoise(predict, sched, (16, 24, 3), "cpu", None, GuideConfig(mode="navdp"),
                             torch.Generator().manual_seed(0))
        torch.testing.assert_close(plain["moves"], ref["moves"])
        # Choosing: tube only among certified samples; navdp/obstacle keep the best value.
        values, certified = np.array([3.0, 2.0, 1.0]), np.array([False, True, True])
        self.assertEqual(choose(values, certified, "tube"), (1, True))
        self.assertEqual(choose(values, certified, "navdp"), (0, False))
        self.assertEqual(choose(values, np.zeros(3, bool), "tube"), (0, False))
        self.assertEqual(len(results["obstacle"][0]["tubes"]), 0)   # no tubes on the finished paths

    def test_agent_wrapper_with_a_stand_in_policy(self):
        from navdp.tube.agent import pointgoal_tube
        from navdp.tube.guided_sampler import GuideConfig
        predict, sched = straight_through()

        class Policy:
            device, predict_size, noise_scheduler = "cpu", 24, sched
            rgbd_encoder = staticmethod(lambda images, depths: torch.zeros(1, 4, 8))
            point_encoder = staticmethod(lambda goal: torch.zeros(goal.shape[0], 8))
            predict_noise = staticmethod(lambda x, k, goal, rgbd: predict(x, k))
            predict_critic = staticmethod(lambda moves, rgbd: -moves[:, :, 1].abs().sum(dim=1))  # prefers straight

        out = pointgoal_tube(Policy(), np.array([[5.5, 0.0, 0.0]]), None, None, rock_bev(), GuideConfig(mode="tube"))
        self.assertEqual(out["candidates"].shape, (16, 24, 3))
        self.assertEqual(len(out["certified"]), 16)
        json.dumps(out["tubes"])  # the server sends it as JSON


class Client(unittest.TestCase):
    def test_sends_the_grid_and_mode(self):
        from navdp.tube.client import TubeNavDPClient
        sent = {}

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"algo": "navdp"} if "reset" in sent["url"] else {
                    "trajectory": [[[0.25, 0.0, 0.0]] * 24], "all_trajectory": [[[[0.25, 0.0, 0.0]] * 24] * 2],
                    "all_values": [[1.0, 0.0]], "tube": {"mode": "tube", "certified": [True, False]}}

        class Session:
            def post(self, url, **kwargs):
                sent["url"], sent["kwargs"] = url, kwargs
                return Response()

            def close(self):
                pass

        client = TubeNavDPClient(session=Session())
        client.reset(np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]))
        rgb, depth = np.zeros((8, 8, 3), np.uint8), np.ones((8, 8), np.float32)
        bev = rock_bev()
        result = client.step(rgb, depth, np.array([3.0, 0.5]), bev, "tube")
        data = json.loads(sent["kwargs"]["data"]["tube_data"])
        self.assertEqual(data["mode"], "tube")
        np.testing.assert_array_equal(Bev.decode(data["bev"]).occupied, bev.occupied)
        self.assertEqual(result.tube["certified"], [True, False])
        self.assertEqual(result.candidates.shape, (2, 24, 3))


class TubeController(unittest.TestCase):
    def test_time_map_is_monotone_and_smooth(self):
        from navdp.tube.controller import FunnelConfig, TubeClock
        tubes, _, _ = tubes_for(detour(1)[None], Bev.empty(), TubeConfig(), push=False)
        clock = TubeClock(tubes[0].lo_coef, tubes[0].hi_coef, 24, FunnelConfig())
        t = np.linspace(0, clock.duration, 2000)
        s = np.array([clock.s_at(v) for v in t])
        self.assertTrue((np.diff(s) >= -1e-12).all())
        self.assertAlmostEqual(s[0], 0.0)
        self.assertAlmostEqual(s[-1], 1.0)
        h = 1e-5                                                       # no kink at an inner knot
        knot = clock.knots_t[5]
        left = (clock.s_at(knot) - clock.s_at(knot - h)) / h
        right = (clock.s_at(knot + h) - clock.s_at(knot)) / h
        self.assertAlmostEqual(left, right, delta=1e-2 * max(abs(left), 1e-6))

    def test_point_stays_inside_a_moving_tube(self):
        from navdp.tube.controller import FunnelConfig, TubeClock, funnel_command
        j = np.arange(1, 25)
        paths = np.stack([np.stack([0.22 * j, np.minimum(1.4, 0.18 * j) + 0.05 * k], 1) for k in range(4)])
        tubes, group, _ = tubes_for(sweep(paths, 0.25), Bev.empty(), TubeConfig(start_forward=0.2), push=False)
        self.assertEqual(len(tubes), 1)
        self.assertTrue(tubes[0].feasible)
        cfg = FunnelConfig()
        clock = TubeClock(tubes[0].lo_coef, tubes[0].hi_coef, 24, cfg)
        pos, heading, dt, outside, saturated = np.zeros(2), 0.0, 1 / 30, 0, 0
        steps = int(clock.duration / dt)
        for i in range(steps):
            lo, hi, velocity = clock.at(i * dt)
            point = pos + cfg.lookahead_m * np.array([math.cos(heading), math.sin(heading)])
            v, omega, info = funnel_command(point, heading, lo, hi, velocity, cfg)
            self.assertLessEqual(v, cfg.max_speed)
            self.assertLessEqual(abs(omega), cfg.max_yaw_rate)
            outside += not info["inside"]
            saturated += info["saturated"]
            pos = pos + dt * v * np.array([math.cos(heading), math.sin(heading)])
            heading += dt * omega
        self.assertEqual(outside, 0)
        self.assertLess(saturated / steps, 0.2)

    def test_law_pushes_back_from_a_wall(self):
        from navdp.tube.controller import FunnelConfig, funnel_command
        cfg = FunnelConfig(max_speed=99, max_yaw_rate=99)
        v, omega, info = funnel_command([0.0, 0.45], 0.0, [-0.5, -0.5], [0.5, 0.5], [0.0, 0.0], cfg)
        self.assertLess(omega, 0)  # near the left wall: turn right
        self.assertTrue(info["inside"])


def world_disc(x, y, r, res=0.05, lo=(-5.0, -5.0), shape=(300, 300)):
    """A world-frame map (row = y, like ROS) with one round blob of occupied cells."""
    from navdp.tube.driver import WorldGrid
    cy = lo[1] + (np.arange(shape[0]) + 0.5) * res
    cx = lo[0] + (np.arange(shape[1]) + 0.5) * res
    gx, gy = np.meshgrid(cx, cy)
    return WorldGrid((gx - x) ** 2 + (gy - y) ** 2 <= r * r, res, lo)


class Driver(unittest.TestCase):
    def test_frames(self):
        from navdp.tube.driver import to_body
        np.testing.assert_allclose(to_body((1.0, 3.0), (1.0, 1.0, math.pi / 2)), [2.0, 0.0], atol=1e-9)  # straight ahead
        np.testing.assert_allclose(to_body((0.0, 1.0), (1.0, 1.0, math.pi / 2)), [0.0, 1.0], atol=1e-9)  # on its left

    def test_world_map_to_body_grid(self):
        from navdp.tube.driver import WorldGrid, body_bev
        occupied = np.zeros((200, 200), bool)                       # x, y in [-5, 5), 5 cm cells
        occupied[180, 100] = True                                   # one cell at world (0.0, 4.0)
        world = WorldGrid(occupied, 0.05, (-5.0, -5.0))
        bev = body_bev(world, (0.0, 0.0, math.pi / 2), grow_m=0.3)  # the robot faces +y: the cell is 4 m ahead
        self.assertTrue(bev.box_blocked([4.02, -0.02], [4.02, -0.02]))
        self.assertTrue(bev.box_blocked([3.75, 0.0], [3.75, 0.0]))   # grown by the robot
        self.assertFalse(bev.box_blocked([3.4, 0.0], [3.4, 0.0]))
        self.assertFalse(bev.box_blocked([0.0, 4.0], [0.0, 4.0]))    # not on its left

    def test_plan_sends_a_body_grid(self):
        from navdp.tube.driver import DriverConfig, TubeDriver
        got = {}

        class Client:
            def reset(self, intrinsic, threshold):
                pass

            def close(self):
                pass

            def step(self, rgb, depth, goal, bev, mode, start_forward=0.0, **_):
                got["bev"] = bev
                path = np.stack([0.2 * np.arange(1, 25), np.zeros(24), np.zeros(24)], 1).astype(np.float32)
                result = NavDPResult(path, {}, path[None], np.zeros(1, np.float32))
                result.tube = {"certified": [False], "chosen_certified": False, "choice": 0, "group": [0],
                               "tubes": [{"solved": False}]}
                return result

        driver = TubeDriver(config=DriverConfig(controller="track"), client=Client())
        driver.plan(None, None, (5.0, 0.0), (0.0, 0.0, 0.0), world_disc(1.5, 0.0, 0.2), now=0.0)
        self.assertIsInstance(got["bev"], Bev)
        self.assertTrue(got["bev"].box_blocked([1.5, 0.0], [1.5, 0.0]))
        driver.plan(None, None, (5.0, 0.0), (0.0, 0.0, 0.0), None, now=0.1)
        self.assertIsNone(got["bev"])

    def test_drives_round_an_obstacle_in_a_plain_world_frame(self):
        from navdp.tube.driver import DriverConfig, TubeDriver

        class Client:
            def reset(self, intrinsic, threshold):
                pass

            def close(self):
                pass

            def step(self, rgb, depth, goal, bev, mode, start_forward=0.0, **_):
                j = np.arange(1, 25)
                forward = 0.22 * j
                lateral = goal[1] * forward / max(goal[0], forward[-1])
                grid = bev if bev is not None else Bev.empty()
                cells = np.argwhere(grid.occupied)
                if len(cells):                                   # bend round the occupied blob, as NavDP would
                    pts = np.asarray(grid.origin) + (cells + 0.5) * grid.resolution
                    of, ol = pts.mean(axis=0)
                    r = np.linalg.norm(pts - [of, ol], axis=1).max()
                    if of > 0.3:
                        bump = np.clip(1.0 - np.abs(forward - of) / (r + 1.2), 0.0, 1.0)
                        lateral = lateral + (ol + r + 0.4) * np.minimum(1.0, 1.6 * bump)
                path = sweep(np.stack([forward, lateral], axis=1), 0.25)
                path = np.concatenate([path, np.zeros((24, 1))], axis=1).astype(np.float32)
                tubes, group, _ = tubes_for(path[None, :, :2], grid, TubeConfig(start_forward=start_forward),
                                            push=False)
                certified = bool(inside(path[None, :, :2], tubes[group[0]])[0])
                result = NavDPResult(path, {}, path[None], np.zeros(1, np.float32))
                result.tube = {"certified": [certified], "chosen_certified": certified, "choice": 0,
                               "group": group.tolist(), "tubes": [t.to_dict(waypoint_times(24)) for t in tubes]}
                return result

        goal, world = (0.0, 8.0), world_disc(0.0, 4.0, 0.75, lo=(-5.0, -2.0), shape=(240, 200))  # facing +y
        for controller in ("stt", "track"):
            driver = TubeDriver(config=DriverConfig(controller=controller), client=Client())
            pose, closest, reached = [0.0, 0.0, math.pi / 2], np.inf, False
            with self.assertRaises(RuntimeError):
                driver.command(pose, 0.0)
            for step in range(800):
                now = step / 10.0
                if step % 2 == 0:
                    driver.plan(None, None, goal, pose, world, now)
                v, omega, info = driver.command(pose, now)
                pose = [pose[0] + v * math.cos(pose[2]) / 10, pose[1] + v * math.sin(pose[2]) / 10, pose[2] + omega / 10]
                closest = min(closest, math.hypot(pose[0], pose[1] - 4.0))
                if math.hypot(pose[0] - goal[0], pose[1] - goal[1]) < 0.6:
                    reached = True
                    break
            self.assertTrue(reached, controller)
            self.assertGreater(closest, 0.75, controller)


if __name__ == "__main__":
    unittest.main()
