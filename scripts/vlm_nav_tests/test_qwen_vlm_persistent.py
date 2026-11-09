"""
Measures per-call latency of the persistent Qwen VLM server.

Usage:
  1. Start the server (in the qwen_vlm env):
       conda activate qwen_vlm && python scripts/vlm_nav_tests/qwen_vlm_server.py
  2. Run this script (stdlib only, any env):
       python scripts/vlm_nav_tests/test_qwen_vlm_persistent.py --rgb vlm_nav_out/rgb_0000.png
"""

import argparse
import time

from qwen_vlm_client import ping, query_vlm_persistent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--hz", type=float, default=3.0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    print("ping:", ping(args.host, args.port))

    period = 1.0 / args.hz
    latencies = []
    for i in range(args.n):
        t0 = time.time()
        text = query_vlm_persistent(
            args.rgb,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            host=args.host,
            port=args.port,
        )
        dt = time.time() - t0
        latencies.append(dt)
        print(f"call {i:02d}: {dt * 1000:6.1f} ms -> {text[:70]!r}")
        remaining = period - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)

    print(
        f"\nn={len(latencies)} mean={1000 * sum(latencies) / len(latencies):.1f}ms "
        f"min={1000 * min(latencies):.1f}ms max={1000 * max(latencies):.1f}ms"
    )


if __name__ == "__main__":
    main()
