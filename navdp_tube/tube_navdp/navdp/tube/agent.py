from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from navdp.tube.guided_sampler import GuideConfig, choose, guided_denoise
from navdp.tube.stt import waypoint_times


def import_official(navdp_root):
    path = str((Path(navdp_root).expanduser() / "baselines" / "navdp").resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from policy_agent import NavDP_Agent  # noqa: E402
    return NavDP_Agent

def pointgoal_tube(policy, goal_point, input_images, input_depths, bev, cfg: GuideConfig, sample_num=16,
                   generator=None, record: bool = False, prev=None) -> dict:
    """bev: a stt.Bev in the body frame, grown by the rover's size (None = no obstacles). prev: the previous plan
    in this body frame (guided_denoise; None = none)."""
    with torch.no_grad():
        goal = torch.as_tensor(goal_point, dtype=torch.float32, device=policy.device)
        rgbd_embed = policy.rgbd_encoder(input_images, input_depths)
        goal_embed = policy.point_encoder(goal).unsqueeze(1)
        rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
        goal_embed = torch.repeat_interleave(goal_embed, sample_num, dim=0)
        out = guided_denoise(lambda x, k: policy.predict_noise(x, k, goal_embed, rgbd_embed), policy.noise_scheduler,
                             (sample_num, policy.predict_size, 3), policy.device, bev, cfg, generator,
                             record, prev)
        values = policy.predict_critic(out["moves"], rgbd_embed).cpu().numpy()
        trajectory = torch.cumsum(out["moves"] / 4.0, dim=1)
        short = trajectory[:, -1, :2].norm(dim=-1) < 0.5  # the official rule: a short path becomes "turn in place"
        trajectory[short] = trajectory[short] * torch.tensor([[0.0, 0.0, 1.0]], device=trajectory.device)
    candidates = trajectory.cpu().numpy()
    index, certified = choose(values, out["certified"], cfg.mode)
    s = waypoint_times(policy.predict_size)
    return {"candidates": candidates, "values": values, "choice": index, "chosen_certified": certified,
            "certified": out["certified"], "group": out["group"],
            "tubes": [tube.to_dict(s) for tube in out["tubes"]],
            "split_paths": None if out["split_paths"] is None else out["split_paths"],
            "x0_steps": out["x0_steps"], "lyapunov_v": out["lyapunov_v"], "tube_builds": out["tube_builds"],
            "model_v": out["model_v"], "final_v": out["final_v"], "final_in_cells": out["final_in_cells"]}


def make_agent(navdp_root, intrinsic, checkpoint, device="cuda:0", stop_threshold=-4.0):
    official = import_official(navdp_root)

    class TubeNavDPAgent(official):
        def step_pointgoal_tube(self, goals, images, depths, bev, cfg: GuideConfig, sample_num=16,
                                record=False, prev=None):
            """The official ``step_pointgoal`` for batch size 1 with the guided sampler."""
            process_images, process_depths = self.process_image(images), self.process_depth(depths)
            queue = self.memory_queue[0]
            if len(queue) >= self.memory_size:
                del queue[0]
            queue.append(process_images[0])
            input_image = np.array(queue)
            input_image = np.pad(input_image, ((self.memory_size - input_image.shape[0], 0), (0, 0), (0, 0), (0, 0)))
            out = pointgoal_tube(self.navi_former, self.process_pointgoal(goals), input_image[None], process_depths,
                                 bev, cfg, sample_num, record=record, prev=prev)
            chosen = out["candidates"][out["choice"]].copy()
            if out["values"].max() < self.stop_threshold:  # the official stop rule: turn toward the mean side
                chosen[:, 0] = 0.0
                chosen[:, 1] = np.sign(chosen[:, 1].mean())
            out["trajectory"] = chosen
            return out

    agent = TubeNavDPAgent(intrinsic, image_size=224, memory_size=8, predict_size=24, temporal_depth=16, heads=8,
                           token_dim=384, navi_model=checkpoint, device=device)
    # The official agent passes device in the position of NavDP_Policy's channels argument, so the policy and its
    # encoders keep their default GPU device whatever was asked for.
    for module in agent.navi_former.modules():
        if isinstance(getattr(module, "device", None), str):
            module.device = device
    agent.reset(1, stop_threshold)
    return agent
