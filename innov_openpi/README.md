# innov_openpi

openpi框架相关代码，包含 SFT 微调、RL Token 训练和在线强化学习。

## 项目组成

- **π₀.₅ VLA 模型**：PyTorch 实现，支持推理与监督微调
- **RL Token**：信息瓶颈编码器-解码器，将 VLA 嵌入压缩为紧凑状态表示
- **在线 RL 训练**：TD3 算法（双 Q-critic + BC 正则化），在线优化策略
- **策略服务**：WebSocket 策略服务器，支持远程推理

## 安装

### 1. 安装依赖

```bash
conda create -n innov_openpi python=3.11
conda activate innov_openpi
pip install -e .
```

### 2. 应用 transformers 补丁

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* $CONDA_PREFIX/lib/python3.11/site-packages/transformers/
```

### 3. 下载模型检查点

```bash
gsutil cp -r gs://openpi-assets/checkpoints/pi05_base ./checkpoints/
```

模型检查点默认缓存到 `~/.cache/openpi`，可通过 `OPENPI_DATA_HOME` 环境变量覆盖。

## 使用

### SFT 微调

```bash
# 计算归一化统计量
python scripts/compute_norm_stats.py --config configs/bi_s1/pi05_finetune.yaml

# 单卡训练
python scripts/train_pytorch.py --config configs/bi_s1/pi05_finetune.yaml --exp_name my_run

# 多卡 DDP
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/train_pytorch.py --config configs/bi_s1/pi05_finetune.yaml --exp_name my_run
```

### Stage 1 — RL Token 训练

```bash
# 冻结 VLA，仅训练 RL Token
python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml \
    --train.vla_checkpoint_dir checkpoints/my_run/20000

# 联合训练（同时微调 VLA 和 RL Token）
python scripts/train_rl_token.py --config configs/rlt/stage1_rl_token.yaml \
    --train.vla_checkpoint_dir checkpoints/my_run/20000 \
    --train.vla_finetune_alpha 0.5
```

### Stage 2 — 在线 RL 训练

```bash
python scripts/train_online_rl.py --config configs/rlt/stage2_online_rl.yaml \
    --vla_checkpoint_dir checkpoints/my_run/20000 \
    --rl_token_checkpoint checkpoints/rl_token/my_run/rl_token_step5000.pt \
    --env_factory my_package.envs.make_env \
    --task_prompt "stack the three blocks"
```

### 推理

```bash
# 启动策略服务器
python scripts/serve_policy.py policy:checkpoint \
    --policy.config=configs/bi_s1/pi05_inference.yaml \
    --policy.dir=checkpoints/my_run/20000

# VLA rollout
python scripts/rollout_vla.py \
    --vla-config-name configs/bi_s1/pi05_finetune.yaml \
    --vla-checkpoint-dir checkpoints/my_run/20000 \
    --env-factory my_package.envs.make_env \
    --task-prompt "pick up the object" \
    --num-episodes 10
```

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| GPU 显存不足 | 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`；或多卡 DDP |
| 缺少 norm stats | 先运行 `scripts/compute_norm_stats.py` |
| 导入错误 | 确认已应用 transformers 补丁 |
| 训练 loss 发散 | 检查 `norm_stats.json` 中 `q01`/`q99`/`std` 值 |
