import unittest
from types import SimpleNamespace

import numpy as np
import torch

from baselines.navdp.tube_planner.depth_obstacles import DepthObstacleBatch
from baselines.navdp.tube_planner.s2diff_guidance import (
    S2DiffGuidanceConfig,
    TrajectoryEnergy,
    sample_gradient_pointgoal_candidates,
    sample_s2diff_pointgoal_candidates,
    smc_particle_mean,
    trajectory_energy,
)


def obstacle_batch(points: list[list[float]]) -> DepthObstacleBatch:
    if points:
        tensor = torch.tensor([points], dtype=torch.float32)
        mask = torch.ones((1, len(points)), dtype=torch.bool)
    else:
        tensor = torch.zeros((1, 1, 2), dtype=torch.float32)
        mask = torch.zeros((1, 1), dtype=torch.bool)
    return DepthObstacleBatch(tensor, mask)


class S2DiffEnergyTest(unittest.TestCase):
    def test_robot_radius_converts_center_distance_to_surface_clearance(self) -> None:
        trajectories = torch.tensor(
            [[[[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]]], dtype=torch.float32
        )
        obstacle = obstacle_batch([[1.0, 0.5]])
        point_result = trajectory_energy(
            trajectories,
            torch.tensor([[2.0, 0.0, 0.0]]),
            obstacle,
            S2DiffGuidanceConfig(
                robot_radius=0.0,
                hard_collision_distance=0.2,
                safe_distance=0.4,
            ),
        )
        rover_result = trajectory_energy(
            trajectories,
            torch.tensor([[2.0, 0.0, 0.0]]),
            obstacle,
            S2DiffGuidanceConfig(
                robot_radius=0.35,
                hard_collision_distance=0.2,
                safe_distance=0.4,
            ),
        )

        self.assertFalse(point_result.collision[0, 0].item())
        self.assertTrue(rover_result.collision[0, 0].item())
        self.assertAlmostEqual(point_result.minimum_clearance[0, 0].item(), 0.5)
        self.assertAlmostEqual(rover_result.minimum_clearance[0, 0].item(), 0.15)

    def test_hard_gibbs_factor_rejects_a_colliding_straight_path(self) -> None:
        trajectories = torch.tensor(
            [
                [
                    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                    [[0.5, 0.5, 0.0], [1.0, 0.7, 0.0], [1.5, 0.5, 0.0]],
                ]
            ],
            dtype=torch.float32,
        )
        result = trajectory_energy(
            trajectories,
            torch.tensor([[2.0, 0.0, 0.0]]),
            obstacle_batch([[1.0, 0.0]]),
            S2DiffGuidanceConfig(
                hard_collision_distance=0.2, safe_distance=0.4
            ),
        )

        self.assertTrue(result.collision[0, 0].item())
        self.assertFalse(result.collision[0, 1].item())
        self.assertGreater(result.total[0, 0].item(), result.total[0, 1].item())

    def test_almost_lyapunov_factor_favors_goal_progress(self) -> None:
        trajectories = torch.tensor(
            [
                [
                    [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                    [[-0.2, 0.0, 0.0], [-0.4, 0.0, 0.0], [-0.6, 0.0, 0.0]],
                ]
            ],
            dtype=torch.float32,
        )
        result = trajectory_energy(
            trajectories,
            torch.tensor([[2.0, 0.0, 0.0]]),
            obstacle_batch([]),
            S2DiffGuidanceConfig(),
        )

        self.assertLess(result.lyapunov[0, 0], result.lyapunov[0, 1])

    def test_smc_mean_uses_safe_particle_without_crossing_candidates(self) -> None:
        particles = torch.zeros((1, 2, 2, 1, 3), dtype=torch.float32)
        particles[0, 0, 0, 0, 0] = 1.0
        particles[0, 0, 1, 0, 0] = -1.0
        particles[0, 1, 0, 0, 1] = 2.0
        particles[0, 1, 1, 0, 1] = -2.0
        soft = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
        collision = torch.tensor([[[False, True], [True, False]]])
        zeros = torch.zeros_like(soft)
        energy = TrajectoryEnergy(
            total=zeros,
            soft_total=soft,
            minimum_clearance=zeros,
            collision=collision,
            safety=zeros,
            lyapunov=zeros,
            terminal_goal=zeros,
            barrier=zeros,
            circulation=zeros,
            mode_switch=zeros,
        )

        mean, weights = smc_particle_mean(particles, energy, temperature=1.0)

        self.assertAlmostEqual(mean[0, 0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(mean[0, 1, 0, 1].item(), -2.0)
        np.testing.assert_allclose(weights.numpy(), [[[1.0, 0.0], [0.0, 1.0]]])

    def test_circulation_sign_prevents_left_right_mode_cancellation(self) -> None:
        path = [[0.2, 0.0, 0.0], [0.4, -0.1, 0.0], [0.6, -0.25, 0.0]]
        trajectories = torch.tensor([[path, path]], dtype=torch.float32)
        result = trajectory_energy(
            trajectories,
            torch.tensor([[2.0, 0.0, 0.0]]),
            obstacle_batch([[1.0, 0.0]]),
            S2DiffGuidanceConfig(
                safe_distance=0.2,
                hard_collision_distance=0.1,
                safety_weight=0.0,
                terminal_goal_weight=0.0,
                lyapunov_weight=0.0,
                nominal_weight=0.0,
                smoothness_weight=0.0,
                step_weight=0.0,
                barrier_weight=0.0,
                circulation_weight=1.0,
                circulation_activation_distance=2.0,
                minimum_circulation_progress=0.05,
                circulation_switch_weight=0.0,
            ),
            circulation_signs=torch.tensor([[1.0, -1.0]]),
        )

        self.assertLess(result.circulation[0, 0], result.circulation[0, 1])


class FakeScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=2)
        self.alphas_cumprod = torch.tensor([0.8, 0.5])
        self.timesteps = torch.tensor([1, 0])

    def set_timesteps(self, _: int) -> None:
        self.timesteps = torch.tensor([1, 0])

    def step(self, model_output, timestep, sample):
        del timestep
        return SimpleNamespace(prev_sample=(sample - 0.1 * model_output).clamp(-1, 1))


class FakePolicy:
    device = "cpu"
    predict_size = 4

    def __init__(self) -> None:
        self.noise_scheduler = FakeScheduler()

    def rgbd_encoder(self, images, depths):
        del depths
        return torch.zeros((images.shape[0], 2, 4))

    def point_encoder(self, goals):
        return torch.zeros((goals.shape[0], 4))

    def predict_noise(self, actions, timestep, goal, rgbd):
        del timestep, goal, rgbd
        return torch.zeros_like(actions)

    def predict_critic(self, *_):
        raise AssertionError("S2Diff must not call the NavDP critic")


class HeadOnScheduler(FakeScheduler):
    def step(self, model_output, timestep, sample):
        del model_output, timestep
        action = torch.zeros_like(sample)
        action[..., 0] = 1.0
        return SimpleNamespace(prev_sample=action)


class HeadOnPolicy(FakePolicy):
    def __init__(self) -> None:
        self.noise_scheduler = HeadOnScheduler()


class S2DiffSamplerTest(unittest.TestCase):
    def test_gradient_sampler_uses_no_particles_and_preserves_shapes(self) -> None:
        result = sample_gradient_pointgoal_candidates(
            FakePolicy(),
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            np.zeros((1, 8, 4, 4, 3), dtype=np.float32),
            np.zeros((1, 4, 4, 1), dtype=np.float32),
            obstacle_batch([]),
            S2DiffGuidanceConfig(
                candidate_count=2,
                particles_per_candidate=1,
                gradient_steps=2,
                gradient_step_size=0.02,
            ),
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(result.selected_trajectory.shape, (1, 4, 3))
        self.assertEqual(result.all_trajectories.shape, (1, 2, 4, 3))
        self.assertEqual(result.energy.shape, (1, 2))
        self.assertEqual(result.diagnostics["particles_per_candidate"], 0)
        self.assertEqual(result.diagnostics["gradient_steps"], 2)
        self.assertFalse(result.fallback_stop[0])

    def test_guided_sampler_preserves_candidate_shapes(self) -> None:
        result = sample_s2diff_pointgoal_candidates(
            FakePolicy(),
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            np.zeros((1, 8, 4, 4, 3), dtype=np.float32),
            np.zeros((1, 4, 4, 1), dtype=np.float32),
            obstacle_batch([]),
            S2DiffGuidanceConfig(candidate_count=3, particles_per_candidate=2),
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(result.selected_trajectory.shape, (1, 4, 3))
        self.assertEqual(result.all_trajectories.shape, (1, 3, 4, 3))
        self.assertEqual(result.energy.shape, (1, 3))
        self.assertFalse(result.fallback_stop[0])
        self.assertFalse(result.escape_turn[0])

    def test_all_colliding_head_on_modes_turn_in_place_instead_of_stopping(self) -> None:
        result = sample_s2diff_pointgoal_candidates(
            HeadOnPolicy(),
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            np.zeros((1, 8, 4, 4, 3), dtype=np.float32),
            np.zeros((1, 4, 4, 1), dtype=np.float32),
            obstacle_batch([[0.5, 0.0]]),
            S2DiffGuidanceConfig(
                candidate_count=2,
                particles_per_candidate=2,
                safe_distance=0.3,
                hard_collision_distance=0.2,
            ),
            generator=torch.Generator().manual_seed(7),
        )

        self.assertTrue(result.escape_turn[0])
        self.assertFalse(result.fallback_stop[0])
        np.testing.assert_allclose(result.selected_trajectory[0, :, 0], 0.0)
        self.assertGreater(np.abs(result.selected_trajectory[0, -1, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
