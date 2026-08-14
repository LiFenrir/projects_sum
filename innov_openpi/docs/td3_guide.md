# TD3 算法解析（基于 innov_openpi 项目）

本文以本项目代码为参照，从零开始梳理 TD3 算法的原理和 RL 核心概念。

---

## 1. 强化学习关键要素

### 1.1 状态 State

智能体感知到的环境信息，是所有决策的依据。

本项目：`x = concat(z_rl, s^p)`，shape `[2056]`。

```python
# rollout_worker.py:101-128
z_rl = rl_token_model.encode(z, pad_mask)   # VLA 嵌入经 encoder 压缩 [2048]
s_p = obs.state[:, :action_dim]              # 本体感知（关节角度 + 夹爪）[8]
x = torch.cat([z_rl, s_p], dim=-1)           # RL 状态 [2056]
```

### 1.2 动作 Action

智能体做出的决策。

本项目：action chunk `[C=10, d=8]`，展平为 80 维向量，控制连续 10 帧的关节速度 + 夹爪开度。

```python
# actor.py:71-72
residual = mlp(cat(x, a_tilde))    # 学习到的残差修正
mu = a_tilde + residual             # VLA 参考 + 修正 → 最终动作
```

### 1.3 策略 Policy

状态到动作的映射 `π(a|x)`，是 RL 要优化的核心对象。

本项目：`Actor` 类（`src/rlt/models/actor.py`）就是策略网络。

```python
# rollout_worker.py:144-158
def _get_actor_action(self, x, a_tilde_flat):
    a_flat = self.actor(x_t, a_tilde_t)    # π(x, a_tilde) → [C*d]
```

### 1.4 奖励 Reward

环境对每一步动作的即时标量反馈。

本项目使用**稀疏奖励**（sparse reward）：中间步骤全为零，只有 episode 终止步有值。

```python
# robot_env.py:148-170
rewards = np.zeros(C, dtype=np.float32)    # 默认全是 0
if human_signal == "s":
    rewards[k] = 1.0   # 成功 → +1
elif human_signal == "f":
    pass               # rewards 保持 0 → 失败 = 0
```

每个 chunk 的 rewards 是形状 `[C]` 的数组，例如：

```
正常执行中:   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
第 4 步成功:  [0, 0, 0, 1.0]  (done=True，提前终止)
```

### 1.5 回报 Return

从当前时刻到 episode 结束的所有奖励的折扣和：

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
```

代码对应（chunk 内的折扣回报）：

```python
# td3_utils.py:51-52
discount_powers = gamma ** torch.arange(C)       # [1, γ, γ², ..., γ⁹]
chunk_return = (rewards * discount_powers).sum()  # Σ γᵏ·rₖ
```

### 1.6 价值函数 Value Function

从某个状态出发按照策略 π 能获得的期望回报：

- **状态价值函数**：`V^π(x) = E[G_t | x_t = x]`
- **动作价值函数**：`Q^π(x, a) = E[G_t | x_t = x, a_t = a]`

本项目使用 Q 函数：

```python
# critic.py:40-50
def forward(self, x, a):
    return self.mlp(torch.cat([x, a], dim=-1))  # → [B, 1] 标量 Q 值
```

Q 网络回答一个问题：**在状态 x 执行动作 chunk a，预计能拿多少回报**。

### 1.7 轨迹 Trajectory

一个 episode 从开始到结束的完整序列：

```
τ = (x₀, a₀, r₁, x₁, a₁, r₂, x₂, ..., x_T)
```

本项目以**转移（transition）**为单位存储在 ReplayBuffer 中：

```python
# replay_buffer.py:47-73
def add(x, a, a_tilde, rewards, next_x, done):
    # 一条转移 = 一个 chunk
    self._x[ptr] = x           # 状态 [2056]
    self._a[ptr] = a           # 动作 [80]
    self._rewards[ptr] = rewards  # 逐帧奖励 [10]
    self._next_x[ptr] = next_x    # 下一状态 [2056]
    self._dones[ptr] = done       # 是否终止 [1]
```

---

## 2. 马尔可夫决策过程 MDP

### 2.1 定义

MDP 是 RL 问题的数学框架，由五元组 `(S, A, P, R, γ)` 定义。核心假设是**马尔可夫性**：下一状态只取决于当前状态和当前动作，与历史无关。

```
P(x_{t+1} | x_t, a_t, x_{t-1}, a_{t-1}, ...) = P(x_{t+1} | x_t, a_t)
```

### 2.2 五元组对应到本项目

| 元素 | 含义 | 本项目 |
|------|------|--------|
| S | 状态空间 | `x ∈ R²⁰⁵⁶` (z_rl + 本体感知) |
| A | 动作空间 | `a ∈ [-1, 1]⁸⁰` (10 帧 × 8 维) |
| P | 状态转移函数 | 机器人 + 物理世界，隐含在 `env.step()` 中 |
| R | 奖励函数 | 稀疏奖励，成功 +1 / 失败 0 |
| γ | 折扣因子 | 0.99 |

### 2.3 交互循环

```
        ┌─── 状态 x ──→ 策略 π ──→ 动作 a ──┐
        │                                      ↓
        │                                   环境 P
        │                                      ↓
        └─── 下一状态 next_x ←── 奖励 r ←─────┘
```

代码对应 `collect_episode` 的主循环：

```python
# rollout_worker.py:228-293
while True:
    x = extract_state(obs)                   # 感知当前状态
    a = actor(x, a_tilde)                    # 策略决策
    next_obs, r, done, info = env.step(a)   # 执行动作，观测转移 + 奖励
    buffer.add(x, a, next_x, r, done)        # 存储经验
    if done: break                           # 终止条件
```

### 2.4 Chunk 级 MDP

本项目不是逐帧决策，而是把连续 10 帧打包成一个决策步（chunk）。这仍是一个合法的 MDP，只是时间粒度更粗：

```
标准 MDP:    x₀ → a₀ → r₀ → x₁ → a₁ → r₁ → x₂ → ...
                γ       γ       γ

Chunk MDP:   x₀ → a[0:10] → Σγᵏrₖ → x₁₀ → a[10:20] → ...
                   γ¹⁰                γ¹⁰
```

---

## 3. 折扣因子 γ

### 3.1 直觉

同样的 100 块钱，**今天拿到**比**一年后拿到**更值钱。RL 里也一样：γ 把未来的奖励"折现"到现在。

### 3.2 数学

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
```

γ=0.99 的折现效果：

| 时间 | 奖励 | 折后价值 |
|------|------|---------|
| 现在 | r=1 | 1.0 |
| 1 步后 | r=1 | 0.99 |
| 10 步后 | r=1 | 0.99¹⁰ ≈ 0.90 |
| 100 步后 | r=1 | 0.99¹⁰⁰ ≈ 0.37 |

### 3.3 代码对应

Chunk 内部逐帧折扣：

```python
# td3_utils.py:51-52
discount_powers = gamma ** torch.arange(10)
# = [1.0, 0.99, 0.9801, 0.9703, 0.9610, 0.9515, 0.9415, 0.9321, 0.9227, 0.9135]
chunk_return = (rewards * discount_powers).sum()
```

跨 chunk bootstrap 用 γ¹⁰：

```python
# td3_utils.py:68
bootstrap = gamma**10 * (1 - done) * Q(next)
#         = 0.904 * Q(next)
```

### 3.4 为什么需要 γ

| 原因 | 说明 |
|------|------|
| 数学收敛 | 无穷级数的折扣和是有限值（几何级数），否则 G 可能无穷大 |
| 鼓励快速完成 | 绕 100 步的奖励远不如 5 步的奖励，策略倾向于缩短路径 |
| 降低远期不确定性 | γ 打折等价于"越远预测越不可信" |

---

## 4. TD3 算法

### 4.1 为什么选 TD3

| 需求 | TD3 如何满足 |
|------|------------|
| 连续动作空间 | 确定性策略梯度，直接输出连续值 |
| Off-policy | buffer 中任意历史数据可反复采样训练，样本效率远高于 PPO |
| 轻量 | Actor/Critic 只是小 MLP（256 维 × 2 层） |

### 4.2 DDPG 的三个问题及 TD3 的解法

标准 DDPG 用同一个 Q 网络既提供 actor 梯度又评估动作。这导致：

| DDPG 问题 | 表现 | TD3 解法 |
|-----------|------|---------|
| **Overestimation bias** | Q 值系统性偏高，actor 追幻觉高 Q 区域 | 双 Q 网络 + 取 min |
| **TD target 方差大** | 同一 (s,a) 的 Q 估计剧烈波动 | 目标策略平滑（加噪声） |
| **Actor-Critic 耦合振荡** | Actor 更新太快，Critic 跟不上 | Actor 延迟更新 |

### 4.3 Q 网络与双 Q 结构

```python
# critic.py:15-50
class QNetwork(nn.Module):
    def forward(self, x, a):
        return self.mlp(torch.cat([x, a], dim=-1))  # [B, 2136] → [B, 1]
```

输入维度：`state_dim + action_chunk_dim = 2056 + 80 = 2136`，输出一个标量 Q 值。

**双 Q 取 min**：

```python
# critic.py:90-97
def q_min(self, x, a):
    q1, q2 = self.forward(x, a)
    return torch.min(q1, q2)   # 选保守估计，抑制 overestimation
```

两个 Q 网络独立初始化、独立训练。取 min 确保：如果其中一个 Q 偏高，min 会选择更保守的那个。

### 4.4 目标网络与 Polyak 平均

```python
# critic.py:110-125
def update_targets(self, tau=0.005):
    # θ_target ← τ·θ_online + (1-τ)·θ_target
    p_target.data.lerp_(p_online.data, tau)
```

目标网络每次只向在线网络挪 0.5%。效果：

- 目标网络变化极慢 → TD target 稳定 → 训练不发散
- 不需要像 DQN 那样每 N 步硬拷贝一次

### 4.5 Actor（策略网络）

```python
# actor.py:58-77
def forward(self, x, a_tilde):
    a_tilde_input = self._apply_ref_dropout(a_tilde)  # 50% 概率置零
    residual = self.mlp(torch.cat([x, a_tilde_input], dim=-1))
    mu = a_tilde + residual                            # 残差结构

    if self.training:
        noise = torch.randn_like(mu) * 0.1             # 探索噪声
        return (mu + noise).clamp(-1.0, 1.0)
    return mu.clamp(-1.0, 1.0)
```

**残差结构 `a = a_tilde + residual`**——Actor 学习 VLA 参考动作的修正量：

- **最后一层零初始化**：初始时 residual=0，Actor 完全复制 VLA → 训练初期就有合理行为
- RL 只优化 VLA 做不好的关键阶段

**参考动作 dropout**：50% 概率将 `a_tilde` 置零，防止 Actor 过度依赖 VLA 参考。

**探索噪声**（σ=0.1）：与环境交互时加的高斯噪声，和 TD3 的目标平滑噪声是两回事：
- 探索噪声 → 收集多样化的训练数据
- 目标平滑噪声 → 稳定 TD target 计算

### 4.6 TD Target 计算

这是 TD3 最核心的公式：

```
y = Σ(k=0→C-1) γᵏ·rₖ  +  γᶜ·(1-done)·min Q_target(x', a')
```

```python
# td3_utils.py:17-70
@torch.no_grad()
def compute_td_target(rewards, dones, next_x, next_a_tilde, actor, critic, gamma, chunk_length):
    # ① Chunk 内折扣回报
    discount_powers = gamma ** torch.arange(C)       # [1, γ, γ², ..., γ⁹]
    chunk_return = (rewards * discount_powers).sum()   # Σ γᵏ·rₖ

    # ② 目标动作 + TD3 平滑噪声
    actor.eval()
    next_a = actor(next_x, next_a_tilde)              # 确定性目标动作
    noise = clip(N(0, 0.2), -0.5, 0.5)               # 目标策略平滑
    next_a = (next_a + noise).clamp(-1, 1)

    # ③ bootstrap
    next_q = critic.target_q_min(next_x, next_a)      # min(Q1_target, Q2_target)
    bootstrap = γᶜ * (1 - done) * next_q

    return chunk_return + bootstrap
```

三个关键设计：

| 步骤 | 作用 |
|------|------|
| 双 Q 取 min | 抑制 overestimation bias |
| 加噪声到目标动作 | 平滑 Q 函数的局部尖峰（Target Policy Smoothing） |
| no_grad + eval | 目标网络不参与梯度计算，TD target 作为常数标签 |

**关于 rₖ**：`rₖ` 是第 k 帧的奖励（标量），不是矩阵的秩。`Σ(k=0→C-1) γᵏ·rₖ` 就是对每帧奖励做折扣加权求和。

### 4.7 Critic 损失

```python
# td3_utils.py:73-90
def critic_loss(q1, q2, q_target):
    return MSE(q1, y) + MSE(q2, y)
```

两个 Q 网络各自向同一个 TD target 回归。

### 4.8 Actor 损失（含 BC 正则）

```python
# td3_utils.py:93-114
def actor_loss(q_value, a, a_tilde, beta=0.5):
    policy_loss = -q_value.mean()
    bc_loss = MSE(a, a_tilde)
    return policy_loss + beta * bc_loss
```

| 项 | 作用 |
|----|------|
| `-Q.mean()` | 最大化 Q 值（RL 目标，取负号变成最小化） |
| `β·MSE(a, a_tilde)` | BC 正则：约束 Actor 不偏离 VLA 参考太远 |

β=0.5 意味着 RL 优化和安全约束各占一半。如果去掉此项，Actor 可能在 Q 网络的盲区（未训练过的动作区域）找到虚假的高 Q 值。

### 4.9 训练循环

```python
# online_rl_trainer.py:118-187
def _update_step(self, update_idx):
    batch = replay_buffer.sample(batch_size=256)

    # ① Critic 更新（每步都做）
    td_target = compute_td_target(...)
    q1, q2 = critic(x, a)
    c_loss = MSE(q1, y) + MSE(q2, y)
    critic_optimizer.step()

    # ② Actor 延迟更新（每 2 步做一次）
    if update_idx % 2 == 0:
        a_actor = actor(x, a_tilde)
        q = critic.q_min(x, a_actor)            # 用在线 Q 的 min
        a_loss = -q.mean() + beta * MSE(a_actor, a_tilde)
        actor_optimizer.step()

    # ③ Polyak 更新目标网络
    critic.update_targets(tau=0.005)
```

每收集一个 episode 后跑 `utd_ratio=5` 次 `_update_step()`。

**延迟更新的作用**：Critic 更新 2 次才轮到 Actor 更新 1 次。让 Critic 先追上去，减少 Actor 基于不准确的 Q 做错误决策。

### 4.10 完整数据流

```
env obs → VLA.extract_embeddings() → z (prefix embeddings)
  → RLTokenModel.encode(z) → z_rl [2048]
  → concat(z_rl, proprioceptive) → x [2056]

Actor(x, a_tilde) → a [80] = VLA 参考 + 残差 + 探索噪声
  → reshape → [10, 8] → 环境执行 → (next_obs, rewards, done)

ReplayBuffer.add(x, a, a_tilde, rewards, next_x, done)

采样 batch(256) →
  compute_td_target → y
  critic_loss → 更新 Q1, Q2
  actor_loss → 更新 Actor（延迟）
  Polyak → 更新 Q_target
```

---

## 5. 本项目 TD3 与标准 TD3 的差异

| 维度 | 标准 TD3 | 本项目 |
|------|---------|--------|
| 动作结构 | 单步 action | Chunk C=10 帧，`action_chunk_dim=80` |
| TD bootstrap | `γ·Q(s', a')` | `γ¹⁰·Q(s', a')`，chunk 内用 `Σγᵏrₖ` |
| Actor 架构 | 从零生成动作 | 残差结构 `a_tilde + residual`，零初始化 |
| 参考动作 | 无 | VLA `a_tilde`，50% dropout 防过拟合 |
| Actor loss | `-Q.mean()` | `-Q.mean() + β·MSE(a, a_tilde)` BC 正则 |
| 探索 | 独立探索噪声 | 探索噪声 + VLA 参考引导 |
| 数据来源 | 纯在线交互 | warmup（VLA-only 预填 buffer）+ 在线 RL |

---

## 6. 关键文件索引

| 文件 | 内容 |
|------|------|
| `src/rlt/models/critic.py` | TwinQCritic：双 Q 网络 + 目标网络 + Polyak 更新 |
| `src/rlt/models/actor.py` | Actor：残差策略 + 参考动作 dropout + 探索噪声 |
| `src/rlt/training/td3_utils.py` | compute_td_target / critic_loss / actor_loss |
| `src/rlt/training/online_rl_trainer.py` | 训练循环：warmup → 采集 episode → TD3 更新 |
| `src/rlt/training/replay_buffer.py` | 经验回放缓冲区 |
| `src/rlt/training/config.py` | TD3 超参：γ, τ, UTD, BC β, target noise 等 |

---

## 7. RL Token：信息瓶颈编码器-解码器（Stage 1）

### 7.1 动机

VLA 的 prefix embeddings `z_{1:M}` 是变长的——M 取决于当前图像的 patch 数和 prompt 长度，不同场景下 M 不同。但 RL 的 Actor/Critic 需要**固定维度**的输入。

RL Token 解决的就是这个问题：**把变长的 VLA 嵌入压缩成一个定长的紧凑状态表示 z_rl**。

```
z_{1:M} ∈ R^{M×2048}, M 可变
        ↓ 编码器（信息瓶颈）
z_rl ∈ R^{2048},      固定长度
```

### 7.2 整体架构

```
         VLA 嵌入 z [B, M, 2048]
                │
    ┌───────────┴───────────┐
    │   Encoder (Transformer)│  ← 追加可学习 e_rl token
    │   z_rl = out[:,-1,:]  │
    └───────────┬───────────┘
                │
         z_rl [B, 2048]
                │
    ┌───────────┴───────────┐
    │   Decoder (Transformer)│  ← teacher-forcing + causal mask
    │   交叉注意力到 z_rl    │
    │   z_hat = h_phi(out)  │
    └───────────┬───────────┘
                │
         z_hat [B, M, 2048]  → L_ro = MaskedMSE(z_hat, z)
```

### 7.3 编码器 Encoder

```python
# rl_token.py:15-75
class RLTokenEncoder(nn.Module):
    def __init__(self, embedding_dim=2048, num_layers=2, num_heads=8):
        self.e_rl = nn.Parameter(torch.randn(1, 1, 2048) * 0.02)  # 可学习 RL token
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, z, pad_mask):
        # 追加 e_rl 到序列末尾: [B, M, D] → [B, M+1, D]
        e_rl = self.e_rl.expand(B, -1, -1)
        tokens = torch.cat([z, e_rl], dim=1)

        # Transformer 编码（e_rl 可 attend 到所有 z_i，pad 位被 mask）
        out = self.transformer(tokens, src_key_padding_mask=~pad_mask)

        # 取最后一个位置 = RL token
        z_rl = out[:, -1, :]   # [B, 2048]
        return z_rl
```

**核心机制**：`e_rl` 是一个可学习的嵌入向量，追加到 VLA 嵌入序列末尾。通过 2 层 Transformer 的自注意力，`e_rl` 可以从所有有效（非 pad）的 z_i 中聚合信息，最终被压缩为单个 token `z_rl`。

**为什么是信息瓶颈**：整个变长序列（M 可达数百）被压缩到一个 2048 维向量。解码器能否重建原始嵌入，决定了 z_rl 保留了多少信息。

### 7.4 解码器 Decoder

```python
# rl_token.py:78-152
class RLTokenDecoder(nn.Module):
    def __init__(self, embedding_dim=2048, num_layers=2, num_heads=8):
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.h_phi = nn.Linear(2048, 2048)   # 输出投影

    def forward(self, z_rl, z, pad_mask):
        # teacher-forcing 输入: [z_rl, z_1, ..., z_{M-1}]
        tgt = torch.cat([z_rl.unsqueeze(1), z[:, :-1, :]], dim=1)  # [B, M, D]

        # causal mask: 位置 i 只能看到 ≤i
        causal_mask = generate_square_subsequent_mask(M)

        # memory: z_rl 作为交叉注意力的 key/value
        memory = z_rl.unsqueeze(1)  # [B, 1, D]

        # 解码
        out = self.transformer(tgt, memory,
                               tgt_mask=causal_mask,
                               tgt_key_padding_mask=pad_mask)
        z_hat = self.h_phi(out)   # [B, M, D]
        return z_hat
```

**三个关键设计**：

| 设计 | 说明 |
|------|------|
| **Teacher forcing** | 输入右移一位 `[z_rl, z_1, ..., z_{M-1}]`，位置 i 的输入是 `z_{i-1}`，预测目标却是 `z_i`——标准自回归训练 |
| **Causal mask** | 防止位置 i 偷看未来的 `z_{i+1}`，保证自回归的因果性 |
| **Cross-attention memory** | decoder 每一层都交叉注意到 `z_rl`（单 token），forced 让 z_rl 成为唯一的跨位置信息通道 |

**为什么解码器需要存在**：编码器本身不需要解码器就能输出 z_rl。但光有编码器没法训练——没有监督信号。解码器提供了**自监督的重构损失**：z_rl 必须包含足够信息让解码器重建整个序列，这迫使编码器学会有效压缩。

### 7.5 损失函数：Masked MSE

```python
# rl_token.py:181-211
def forward(self, z, pad_mask):
    z = z.detach()                        # 截断 VLA 梯度
    z_rl = self.encoder(z, pad_mask)
    z_hat = self.decoder(z_rl, z, pad_mask)

    mse = (z_hat - z).pow(2).mean(dim=-1)       # [B, M]
    masked_mse = mse * pad_mask.float()           # pad 位置置零
    loss = masked_mse.sum() / pad_mask.sum()      # 仅有效位置求均值
    return loss, z_rl, z_hat
```

**Mask 的作用**：不同样本的 M 不同（图像 patch 数 + prompt 长度）。用 pad_mask 标记有效位置，确保短序列的 padding 不参与 loss 计算。

### 7.6 两种训练模式

| 模式 | α | VLA 状态 | 损失函数 | 用途 |
|------|---|---------|---------|------|
| 冻结 VLA | 0 | freeze，仅提取嵌入 | `L = L_ro` | 默认模式，VLA 不动 |
| 联合训练 | >0 | unfreeze，参与微调 | `L = L_ro + α·L_vla` | VLA 和 RL Token 一起适应下游任务 |

```python
# rl_token_trainer.py:401-460
def _step_joint(self, vla, observations, actions):
    # 单次 VLA 前向：同时拿到 detached z 和 VLA 的 flow-matching loss
    z, pad_mask, l_vla = vla.compute_vla_loss_with_embeddings(observations, actions)

    # L_ro：z 已经 detach → 梯度只流向 encoder-decoder
    l_ro, z_rl, z_hat = self.model(z, pad_mask)

    # 联合损失，但梯度是解耦的
    total_loss = l_ro + alpha * l_vla
    total_loss.backward()
    # l_ro 的梯度 → encoder-decoder
    # l_vla 的梯度 → VLA（通过 forward_with_prefix_embeddings 的非 detach 路径）
```

### 7.7 数据流（Stage 1 → Stage 2 如何衔接）

```
Stage 1 (训练):
  示范数据 → VLA.extract_embeddings() → z [B, M, 2048]
    → Encoder → z_rl → Decoder → z_hat
    → L_ro = MaskedMSE(z_hat, z) → 更新 encoder/decoder

Stage 2 (推理):
  env obs → VLA.extract_embeddings() → z [1, M, 2048]
    → Encoder.encode(z) → z_rl [1, 2048]    ← 冻结，只走 encoder
    → concat(z_rl, s^p) → x [1, 2056]       ← RL 状态
    → Actor(x, a_tilde) → 动作 chunk
```

Stage 1 训练编码器，Stage 2 只使用编码器（`encode()` 方法，`@torch.no_grad()`）。解码器在 Stage 2 中完全不参与——它只是 Stage 1 训练时的"脚手架"。

### 7.8 关键文件索引

| 文件 | 内容 |
|------|------|
| `src/rlt/models/rl_token.py` | RLTokenEncoder / RLTokenDecoder / RLTokenModel |
| `src/rlt/training/rl_token_trainer.py` | Stage 1 训练循环：冻结/联合模式 + LR schedule |
| `src/rlt/training/config.py` | Stage 1 超参：embedding_dim, layers, heads, alpha 等 |
| `src/openpi/training/vla_wrapper.py` | VLAWrapper: extract_embeddings, compute_vla_loss_with_embeddings |

### 7.9 YAML 配置

项目有 4 个预置配置，覆盖不同训练策略：

**`stage1_rl_token.yaml`** — 默认冻结 VLA，5000 步：

```yaml
train:
  embedding_dim: 2048
  encoder_layers: 2        # Transformer encoder 层数
  encoder_heads: 8
  decoder_layers: 2        # Transformer decoder 层数
  decoder_heads: 8

  num_train_steps: 5000
  batch_size: 32
  peak_lr: 1.0e-4
  warmup_steps: 500
  decay_steps: 5000
  decay_lr: 1.0e-5
  max_grad_norm: 1.0

  vla_finetune_alpha: 0.0          # 0 = 冻结 VLA
  vla_learning_rate: 1.0e-5
  gradient_checkpointing: true

  vla_checkpoint_dir: ""            # 必填：SFT 检查点路径
  save_dir: checkpoints/rl_token
  save_every: 1000

repo_id: local/stack_the_blocks     # LeRobot 数据集
num_workers: 4
```

**`stage1_frozen_20k.yaml`** — 延长训练至 20000 步，降低重构误差：

```yaml
train:
  num_train_steps: 20000
  warmup_steps: 1000
  decay_steps: 20000
  vla_finetune_alpha: 0.0            # 冻结
  vla_checkpoint_dir: checkpoints/bi_s1_pi05_sft_shifted/.../24000
  run_name: bi_s1_frozen_20k
repo_id: /home/kemove/VLA/datasets/bi_s1/bi_s1_sft_shifted
```

**`stage1_joint.yaml`** — 联合训练，VLA 与 RL Token 同步优化：

```yaml
train:
  num_train_steps: 10000
  warmup_steps: 500
  decay_steps: 10000
  vla_finetune_alpha: 0.5            # L = L_ro + 0.5 * L_vla
  vla_checkpoint_dir: checkpoints/bi_s1_pi05_sft_shifted/.../24000
  run_name: bi_s1_joint
repo_id: /home/kemove/VLA/datasets/bi_s1/bi_s1_sft_shifted
```

**`stage1_rl_token_bi_s1_sft_shifted.yaml`** — 使用 bi_s1 SFT 检查点的默认配置，与 `stage1_rl_token.yaml` 参数一致，仅检查点和数据集路径不同。

### 7.10 超参解读

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `embedding_dim` | 2048 | 与 VLA prefix embedding 维度一致，**不可修改** |
| `encoder_layers` | 2 | Encoder Transformer 层数，决定压缩能力 |
| `decoder_layers` | 2 | Decoder Transformer 层数，决定重建能力 |
| `peak_lr` | 1e-4 | 峰值学习率，warmup 阶段线性增长至此值 |
| `warmup_steps` | 500 | LR 从 `peak_lr/(warmup+1)` 线性增长到 `peak_lr` 的步数 |
| `decay_steps` | 5000 | Cosine 衰减总步数，`decay_lr` 为终点 |
| `max_grad_norm` | 1.0 | 全局梯度裁剪阈值 |
| `vla_finetune_alpha` | 0.0 | 联合训练权重：0=冻结，>0=联合微调 |
| `batch_size` | 32 | 每步样本数 |
