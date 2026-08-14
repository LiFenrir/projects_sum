"""Shared checkpoint loading utilities."""

from __future__ import annotations

import logging

import torch

from rlt.models.rl_token import RLTokenModel
from rlt.training.config import migrate_rl_token_config

log = logging.getLogger(__name__)


def load_rl_token_model(
    ckpt_path: str,
    device: str = "cuda",
) -> RLTokenModel:
    """从 Stage 1 checkpoint 加载 RL token 模型。

    架构参数从 checkpoint 中保存的 config 读取，支持新旧两种格式。

    Args:
        ckpt_path: Stage 1 ``.pt`` checkpoint 路径。
        device: 加载到的 torch 设备。

    Returns:
        冻结的 ``RLTokenModel``。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_config = migrate_rl_token_config(ckpt["config"])
    model = RLTokenModel(
        embedding_dim=saved_config.arch.embedding_dim,
        encoder_layers=saved_config.arch.encoder_layers,
        encoder_heads=saved_config.arch.encoder_heads,
        decoder_layers=saved_config.arch.decoder_layers,
        decoder_heads=saved_config.arch.decoder_heads,
    )
    model.load_state_dict(ckpt["model"])
    step = ckpt["step"]
    del ckpt
    model = model.to(device)
    log.info("Loaded RL token model from %s (step %d)", ckpt_path, step)
    return model
