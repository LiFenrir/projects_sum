"""Serve a policy via WebSocket for robodeploy deployment.

Loads a trained VLA checkpoint and starts a WebSocket policy server.

Usage:
    # Serve from a YAML config file:
    python scripts/serve_policy.py \\
        --config configs/bi_s1/pi05_inference.yaml \\
        --dir /path/to/checkpoint \\
        --port 8000

    # With default prompt:
    python scripts/serve_policy.py \\
        --config configs/bi_s1/pi05_inference.yaml \\
        --dir checkpoints/my_run/10000 \\
        --default-prompt "pick up the red block" \\
        --port 8000
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.rtc.configuration_rtc import RTCConfig
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Path to YAML config file (e.g., "configs/bi_s1/pi05_inference.yaml").
    config: str = "configs/bi_s1/pi05_inference.yaml"

    # Checkpoint directory (e.g., "checkpoints/bi_s1_pi05_sft/exp/10000").
    dir: str = ""

    # If provided, will be used in case the "prompt" key is not present in the data.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Number of action steps to return per inference. Defaults to 50 for RTC buffering.
    action_chunk: int | None = 50

    # Enable Real-Time Chunking (RTC) for temporal action smoothing.
    # When enabled, the server expects clients to send prev_chunk_left_over /
    # inference_delay / execution_horizon in the RTC protocol envelope.
    rtc: bool = False

    # RTC constraint window (= smooth_window) — guidance overlap + client blend steps.
    rtc_execution_horizon: int = 13

    # Record the policy's behavior for debugging.
    record: bool = False


def main(args: Args) -> None:
    if not args.dir:
        raise ValueError(
            "--dir is required. Provide the path to a checkpoint directory "
            "containing the model weights (e.g., model.safetensors)."
        )

    train_config = _config.load_config(args.config)
    logging.info("Creating policy: config=%s, checkpoint=%s", args.config, args.dir)

    # --- RTC 配置注入 ---
    if args.rtc:
        rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.rtc_execution_horizon,
        )
        # Pi0Config 是 frozen dataclass，需用 object.__setattr__ 绕过（与 create_trained_policy 一致）
        object.__setattr__(train_config.model, "rtc_config", rtc_config)
        logging.info(
            "RTC enabled: execution_horizon=%d", args.rtc_execution_horizon,
        )

    policy = _policy_config.create_trained_policy(
        train_config, args.dir, default_prompt=args.default_prompt, action_chunk=args.action_chunk
    )

    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
