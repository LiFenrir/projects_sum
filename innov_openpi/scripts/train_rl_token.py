"""Stage 1: Train the RL token encoder-decoder on demonstration data.

Trains the information-bottleneck encoder-decoder that compresses
variable-length VLA prefix embeddings z_{1:M} into a single RL token
z_rl via masked-MSE reconstruction loss.

Two modes (selected by ``--train.vla-finetune-alpha``):

- **Frozen VLA** (alpha=0): Only trains the encoder-decoder (phi).
- **Joint training** (alpha>0): Also fine-tunes the VLA (theta) with
  flow-matching loss.  L = L_ro(phi) + alpha * L_vla(theta).

The data pipeline delegates entirely to OpenPI's transform chain so
that normalisation, camera layout, and action chunking exactly match
the pretrained model.

Usage::

    # Single GPU / CPU:
    python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml

    # Multi-GPU DDP (single node):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \\
        scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml

    # Multi-Node DDP:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
        --master_addr=<ip> --master_port=<port> \\
        scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml

    # YAML + CLI overrides:
    python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml \\
        --train.vla-finetune-alpha 1.0

    # Pure CLI (no config file):
    python scripts/train_rl_token.py \\
        --train.vla-checkpoint-dir /path/to/model.safetensors \\
        --repo-id local/stack_the_blocks
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
import sys
from pathlib import Path

from rlt.training.config import RLTokenTrainConfig
from rlt.training.ddp_utils import cleanup_ddp, is_main_process, setup_ddp
from openpi.training.rl_data_loader import build_data_loader
from rlt.training.rl_token_trainer import RLTokenTrainer
from rlt.utils.config_loader import load_config_with_cli
from rlt.utils.logging import Logger
from openpi.training.vla_wrapper import VLAWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger().setLevel(logging.INFO)  # ensure root level even if basicConfig was a no-op
logging.captureWarnings(True)  # route warnings to logging so they appear in run.log
log = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainConfig:
    """Top-level config for Stage 1 training.

    Wraps :class:`RLTokenTrainConfig` (architecture + training hypers)
    and adds dataset / data-transform settings that live outside the
    trainer.
    """

    train: RLTokenTrainConfig = dataclasses.field(default_factory=RLTokenTrainConfig)
    """RL token trainer hyperparameters."""

    repo_id: str = "local/stack_the_blocks"
    """LeRobot dataset repo ID (local or HuggingFace)."""

    data_transforms_fn: str | None = None
    """Dotted import path to a ``(ModelConfig) -> transforms.Group``
    factory that overrides the OpenPI config's default data transforms.
    Example: ``rlt_openpi.policies.franka.config.three_camera_droid``."""

    num_workers: int = 4
    """DataLoader worker processes."""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_data_transforms(dotted_path: str | None, openpi_config_name: str):
    """Dynamically import and call a data-transforms factory."""
    if dotted_path is None:
        return None

    from openpi.training.config import get_config

    module_path, func_name = dotted_path.rsplit(".", 1)
    factory_fn = getattr(importlib.import_module(module_path), func_name)
    return factory_fn(get_config(openpi_config_name).model)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main(config: TrainConfig) -> None:
    # Set up DDP (no-op when WORLD_SIZE=1)
    use_ddp, local_rank, device = setup_ddp()
    main = is_main_process()

    if main:
        # Set up file logging BEFORE any log messages so they all go to run.log
        log_dir = Path(config.train.checkpoint.save_dir) / config.train.checkpoint.run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        # ── Tee stdout/stderr to run.log (captures tqdm, print, everything) ──
        log_file = open(log_dir / "run.log", "w")

        class _TeeWriter:
            """Duplicate writes to a file and the original stream.

            ``\\r`` resets the file line buffer so tqdm progress bars
            don't leave duplicate lines in the log file.
            """

            def __init__(self, file, stream):
                self.file = file
                self.stream = stream
                self._line_buf = ""

            def write(self, data):
                self.stream.write(data)
                for ch in data:
                    if ch == "\r":
                        self._line_buf = ""
                    elif ch == "\n":
                        self.file.write(self._line_buf + "\n")
                        self._line_buf = ""
                    else:
                        self._line_buf += ch
                self.file.flush()

            def flush(self):
                self.file.flush()
                self.stream.flush()

            def isatty(self):
                return self.stream.isatty()

        sys.stdout = _TeeWriter(log_file, sys.__stdout__)
        sys.stderr = _TeeWriter(log_file, sys.__stderr__)

        # Structured logging to the same file handle (preserves timestamps/levels)
        file_handler = logging.StreamHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

        log.info("Log file: %s", log_dir / "run.log")
        log.info("Stage 1 config: %s", config)
        log.info("DDP: %s, device: %s", "enabled" if use_ddp else "disabled", device)

    data_transforms = _resolve_data_transforms(
        config.data_transforms_fn, config.train.checkpoint.vla_config_name
    )

    # VLA is loaded only on the main process's device; its weights are
    # never updated in frozen mode, and in joint mode gradients flow
    # through the VLA directly (the VLA is NOT DDP-wrapped).
    if main:
        log.info(
            "Loading VLA: config=%s, checkpoint=%s",
            config.train.checkpoint.vla_config_name,
            config.train.checkpoint.vla_checkpoint_dir,
        )
    vla = VLAWrapper(
        checkpoint_path=config.train.checkpoint.vla_checkpoint_dir,
        config_name=config.train.checkpoint.vla_config_name,
        device=device,
        data_transforms=data_transforms,
    )

    # Create trainer with DDP support
    trainer = RLTokenTrainer(config.train, device=device, use_ddp=use_ddp)
    rl_logger = Logger.from_train_config(config.train) if main else None

    # Build data loader.  When DDP is active, the underlying
    # torch DataLoader will be created without a DistributedSampler
    # (the rl_data_loader uses a persistent _InfiniteLoader); gradient
    # synchronization is handled by DDP's backward hook automatically.
    if main:
        log.info("Loading demo dataset: %s", config.repo_id)
    data_loader = build_data_loader(
        openpi_config_name=config.train.checkpoint.vla_config_name,
        repo_id=config.repo_id,
        batch_size=config.train.training.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        data_transforms=data_transforms,
    )

    trainer.train(vla, iter(data_loader), log_fn=rl_logger.log if rl_logger else None)

    if rl_logger is not None:
        rl_logger.finish()

    cleanup_ddp()


if __name__ == "__main__":
    # Support --config <yaml_path> with optional CLI overrides
    argv = sys.argv[1:]
    yaml_path = None
    if "--config" in argv:
        idx = argv.index("--config")
        yaml_path = argv[idx + 1]
        del argv[idx : idx + 2]

    config = load_config_with_cli(TrainConfig, yaml_path=yaml_path, cli_args=argv)
    main(config)
