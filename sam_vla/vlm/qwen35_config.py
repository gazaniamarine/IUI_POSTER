"""
Lightweight constants shared between qwen35_server (runs in the qwen_vlm env,
needs transformers) and qwen35_server_manager/qwen35_client (may run in the
habitat env, must NOT need transformers). Keep this module free of heavy
imports.

Distinct port from every other server this repo can have resident at once.
The full known registry (checked 2026-08-24, `grep -rn "PORT\s*=\s*8"`):
8765 = qwen_config.QWEN_SERVER_PORT (Qwen2.5-VL-3B), 8766 = BOTH
navdp_upstream_server_manager._DEFAULT_PORT (real upstream NavDP) AND
vl_direction.config.INTERNVL_SERVER_PORT (those two only coexist safely
because they're never both running in the same process), 8767 =
vl_direction.config.QWEN_SERVER_PORT, 8768 = qwen_ablation_server's
default, 8790/8791 = the ablation harness's ports. 8766 in particular is
NOT safe to reuse here: nav.rover_controller runs NavdpUpstreamPolicy (port
8766) and this Qwen3.5-9B server in the very same process (see
start_vlm_navigation), unlike vl_direction's reuse of that same number
elsewhere. 8769 is the first number after all of the above.
"""

QWEN35_SERVER_HOST = "127.0.0.1"
QWEN35_SERVER_PORT = 8769
