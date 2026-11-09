#!/usr/bin/env python3
"""Measure the isolated cost of NavDP's learned critic decoder pass."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navdp-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    navdp_root = Path(args.navdp_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    module_root = navdp_root / "baselines" / "navdp"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not module_root.is_dir():
        raise FileNotFoundError(module_root)
    sys.path.insert(0, str(module_root))

    from policy_agent import NavDP_Agent

    intrinsic = np.eye(3, dtype=np.float32)
    agent = NavDP_Agent(
        intrinsic,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        navi_model=str(checkpoint),
        device=args.device,
    )
    policy = agent.navi_former
    device = torch.device(args.device)
    actions = torch.randn(
        args.candidates, policy.predict_size, 3, device=device
    )
    rgbd = torch.randn(
        args.candidates,
        policy.memory_size * 16,
        policy.token_dim,
        device=device,
    )

    critic_parameters = sum(parameter.numel() for parameter in policy.critic_head.parameters())
    critic_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in policy.critic_head.parameters()
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            policy.predict_critic(actions, rgbd)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                policy.predict_critic(actions, rgbd)
            end.record()
            torch.cuda.synchronize(device)
            total_seconds = start.elapsed_time(end) / 1000.0
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            start_time = time.perf_counter()
            for _ in range(args.iterations):
                policy.predict_critic(actions, rgbd)
            total_seconds = time.perf_counter() - start_time
            peak_memory = 0

    result = {
        "device": str(device),
        "candidates": args.candidates,
        "iterations": args.iterations,
        "critic_head_parameters": critic_parameters,
        "critic_head_parameter_bytes": critic_parameter_bytes,
        "critic_decoder_pass_mean_ms": total_seconds * 1000.0 / args.iterations,
        "critic_decoder_passes_per_original_plan": 1,
        "estimated_latency_saved_by_bypassing_critic_ms": (
            total_seconds * 1000.0 / args.iterations
        ),
        "peak_cuda_memory_bytes": int(peak_memory),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
