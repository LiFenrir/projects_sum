# project_sum

项目代码与配套笔记汇总备份(2026-08-14)。

## 目录结构

```
project_sum/
├── robodeploy/          # 【自有】fork 自 LeRobot 的真机部署与数据采集工具包
├── innov_openpi/        # 【自有】fork 自 OpenPI 的 π₀.₅ VLA 训练与在线 RL
├── RISE/                # 【复现】OpenDriveLab 组合世界模型 + 想象中 RL 自改进策略
├── sam3_repo/           # 【复现】SAM3 微调管线(手部/前臂检测)
├── Inpaint-Anything/    # 【复现】逐帧检测 + DiffuEraser 视频修复
└── notes/               # 配套笔记(与项目一一对应)
    ├── robodeploy.md                        # 项目介绍:注册机制、环境安装、功能使用
    ├── innov-openpi.md                      # 项目介绍:LoRA/全参微调、RLT、RECAP 配置指南
    ├── rise.md                              # 复现指南:离线策略/价值 → 动力学 → 在线 RL
    └── sam3-inpaint-anything-video-inpainting.md  # 复现总结:环境搭建到端到端修复
```

## 说明

- **大文件已排除**:checkpoints、pretrained_models、weights、wandb、.venv、数据集、编译产物(`*.pt/*.pth/*.safetensors/*.mp4` 等)未包含在副本中
- 各仓库已提交保存所有更改(最后一个 commit 为 `chore: 备份前保存未提交更改`)
- 权重/数据恢复方式见各仓库 README 与 notes/ 中对应笔记

## 上传到 GitHub

```bash
cd project_sum/<repo>
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main   # 或 master,以 git branch 显示为准
```
