"""
Lightweight constants shared between qwen_server (runs in the qwen_vlm env,
needs transformers) and qwen_server_manager (runs in the habitat env, must
NOT need transformers). Keep this module free of heavy imports.
"""

QWEN_SERVER_HOST = "127.0.0.1"
QWEN_SERVER_PORT = 8765
