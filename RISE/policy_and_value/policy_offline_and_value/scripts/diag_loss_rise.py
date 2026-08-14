"""RISE 侧诊断:同一权重、同一数据,分解 u_t/v_t/loss(临时,用完删除)"""
import sys
import dataclasses

sys.path.insert(0, "/home/kemove/INNOV/projects/RISE/policy_and_value/policy_offline_and_value/src")

import jax
import safetensors.torch
import torch

import openpi_value.models_pytorch.pi0_pytorch
import openpi_value.training.config as _config
import openpi_value.training.data_loader as _data


def main():
    config = _config._CONFIGS_DICT["LeRobot_pi05_finetune"]
    config = dataclasses.replace(config, num_workers=0)

    # 先建数据 loader,再初始化模型(与 RISE train_pytorch.py 顺序一致)
    loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    for obs, actions in loader:
        break
    print("取到第一个 batch:", tuple(actions.shape), flush=True)

    model_cfg = config.model
    object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    model = openpi_value.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to("cuda")
    safetensors.torch.load_model(
        model,
        "/home/kemove/INNOV/models/openpi/pi05_base_pytorch/model.safetensors",
        strict=False,
    )
    print("权重加载成功", flush=True)

    obs = jax.tree.map(lambda x: x.to("cuda"), obs)
    actions = actions.to("cuda").float()
    print(f"actions: std={actions.std():.4f} mean={actions.mean():.4f}", flush=True)

    with torch.no_grad():
        # 手动复现 forward 前半段
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)
        noise = model.sample_noise(actions.shape, actions.device)
        time = model.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        print(f"time: min={time.min():.4f} max={time.max():.4f}", flush=True)
        print(
            f"noise: std={noise.std():.4f} | x_t: std={x_t.std():.4f} | u_t: std={u_t.std():.4f} "
            f"E[||u||^2]={(u_t**2).mean():.4f}",
            flush=True,
        )

        # 模型前向(eval 模式)
        model.eval()
        print("开始 forward...", flush=True)
        loss = model(obs, actions, noise=noise, time=time)
        print(f"eval 模式 loss.mean(): {loss.mean():.4f}  (B,AH)={tuple(loss.shape)}", flush=True)

        # 手动提取 v_t:复刻 forward 主体
        from openpi_value.models_pytorch import pi0_pytorch as _pi0

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(state, x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = _pi0.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = model._prepare_attention_masks_4d(att_2d_masks)
        if (
            model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        (_, suffix_out), _ = model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = suffix_out[:, -config.model.action_horizon :].to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out)
        print(f"v_t: std={v_t.std():.4f} mean={v_t.mean():.4f}", flush=True)
        cos = torch.mean(
            torch.nn.functional.cosine_similarity(u_t.flatten(0, 1), v_t.flatten(0, 1), dim=1)
        )
        print(f"corr(u_t, v_t): {cos:.4f}", flush=True)
        mse = torch.nn.functional.mse_loss(u_t, v_t, reduction="none")
        print(f"手动 MSE(v_t, u_t): {mse.mean():.4f}", flush=True)
        print(
            f"前14维 loss={mse[:, :, :14].mean():.4f} | 后18维 loss={mse[:, :, 14:].mean():.4f}",
            flush=True,
        )
        print(
            f"E[||v_t||^2]={(v_t**2).mean():.4f} E[||u_t||^2]={(u_t**2).mean():.4f} "
            f"E[v_t*u_t]={(v_t * u_t).mean():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
