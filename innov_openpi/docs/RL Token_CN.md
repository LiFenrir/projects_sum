---
title: "RL Token Bootstrapping Online RL with Vision-Language-Action Models (中文翻译)"
---

# RL Token：基于视觉-语言-动作模型的在线强化学习引导

Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke — Physical Intelligence

https://pi.website/research/rlt

![](images/d80a12a790bfb47d27e6a3774e399f842b90e4275f6f1edf026343ecf2054dcc.jpg)
**图 1：** 我们的方法通过训练一个编码器和解码器，从 VLA 的内部特征中提取紧凑且有意义的表示，将"RL token"引入 VLA。提取的表示随后用于训练轻量级的 actor-critic 网络，并通过样本高效的在线 RL 进行微调，使得高精度任务能够在几小时甚至几分钟的机器人经验中得到优化。

**摘要** — 视觉-语言-动作（VLA）模型可以"开箱即用"地学习执行多种操作技能，但要达到现实世界任务所要求的精度和速度，还需要进一步微调——例如通过强化学习（RL）。我们引入了一种轻量级方法，仅需数小时的真实世界练习即可实现预训练 VLA 的样本高效在线 RL 微调。我们 (1) 适配 VLA 以暴露一个"RL token"——一种紧凑的读出表示，既保留了与任务相关的预训练知识，又可作为在线 RL 的高效接口；(2) 在该 RL token 上训练一个小型的 actor-critic 头来精化动作，同时将学习到的策略锚定到 VLA 上。基于 RL token 的在线 RL（RLT）使得即使是大型 VLA 也能快速高效地进行 RL 微调。在四个真实机器人任务（螺丝安装、扎带紧固、充电器插入和以太网插入）中，RLT 将任务最难部分的速度提升了高达 $3 \times$，并在几分钟到几小时的练习内显著提高了成功率。在某些任务上，其速度甚至超越了人类遥操作。

# I. 引言

通用视觉-语言-动作（VLA）模型可以从数据中学习各种操作技能。然而，它们在执行的"最后一毫米"往往表现挣扎：动作可能缓慢，成功完成可能需要暂停和重试，精确任务关键阶段的微小错误可能累积为失败。解决这一挑战的自然方法是通过强化学习（RL）微调 VLA。通过在目标任务上练习，RL 可以精确改进那些对成功最关键的任务阶段——这些阶段通常对微小错误最敏感，也最难以仅靠演示可靠覆盖。但真实世界的机器人操作面临严格的预算约束：每次试验花费时间，每次失败消耗精力和磨损，有意义的适应性通常必须在几小时的练习内完成。

然而，VLA 的样本高效微调面临重大挑战。一方面，基础模型的传统 RL 训练方法 [1–3] 依赖大规模数据，对快速在线适应来说可能效率低下。另一方面，数据高效的真实世界 RL 方法 [4, 5] 通常训练小得多的模型，虽能在数小时内改进，但牺牲了 VLA 的泛化能力。因此，核心问题是：如何在利用 VLA 泛化能力的同时，实现轻量级在线 RL 的速度和样本效率？

我们提出了一种实用的方案，利用从预训练 VLA 策略中获得的表示来引导快速在线强化学习。我们的核心思想是适配 VLA，使其暴露一个紧凑的接口，用于样本高效的在线 RL。我们通过训练 VLA 暴露一个 **RL token**——一种压缩表示，使任务相关的预训练知识可供轻量级在线 RL 策略访问。使用该 RL token 运行 RL（RLT）创造了简单的分工：冻结的 VLA 提供广泛的感知理解和动作建议，而轻量级的 actor 和 critic 在线地调整策略以成功完成最难的任务部分。为了在样本高效的真实世界场景中实用化，我们的方法使用样本高效的在线 RL 算法来训练使用 RL token 表示的小型 actor-critic 网络，并加入额外的正则化项将 actor 锚定到 VLA 动作上，使得在线 RL 精化有前景的行为而非从头学习。

我们在四个具有挑战性的机器人操作任务上评估了 RLT，这些任务需要毫米或亚毫米级精度：螺丝安装、扎带紧固、以太网和充电器插入。在这些任务中，RLT 在数小时的在线训练内同时提高了成功率和执行速度。最大的收益出现在需要高精度并决定任务成败的关键阶段——RLT 将执行速度提升了高达 $3 \times$，并大幅提高了成功率，例如将一个具有挑战性的螺丝插入任务的成功率从 $20\%$ 提高到 $65\%$。在我们任务中最灵巧的环节之一，用我们的方法训练的策略可以在保持可靠性的同时超越专家遥操作速度。这些结果表明，将 VLA 模型与轻量级在线 RL 相结合，为无需大量任务特定工程即可实现高性能操作提供了一条实用路径。

# II. 相关工作

**视觉-语言-动作模型。** 基于大规模演示数据集的行为克隆近来已成为训练通用机器人操作策略的主导范式（参见 [6–11]）。促成这一成功的两个关键要素是：动作分块（action chunking）[12]——预测多个动作进行顺序开环执行；以及使用表达能力强的输出分布（如扩散模型 [13] 或自回归生成 [6]），能够捕捉演示数据中固有的多模态性。进一步的进展来自使用大型预训练视觉-语言模型作为语言条件通用策略的主干，产生了视觉-语言-动作（VLA）模型 [6, 7]。这些模型将大规模网络先验知识导入到闭环机器人策略中。最近的工作将 VLA 主干与分块动作生成相结合——通过扩散 [8] 或自回归分词 [14, 15]——实现了最先进的通用操作。虽然这些策略展现出令人印象深刻的泛化能力 [9, 16]，但它们在任意给定任务上的性能最终受限于其训练所用的遥操作数据的质量和覆盖范围——当演示本身充满噪声或不一致时，在精度关键型任务上实现可靠的鲁棒成功仍然困难。

**真实世界强化学习。** 强化学习提供了一种超越演示数据性能上限的自然途径：通过在任务上练习，智能体可以发现更快、更精确或更鲁棒的策略，而这些策略从未被演示过。在实践中，机器人真实世界 RL 在严格的样本预算下运行，因为每次机器人 rollout 都消耗时间和磨损。离策略 actor-critic 方法（如 [17–20]）通过重用存储在回放缓冲区中的转移样本来解决这一问题，并且可以通过提高更新-数据比 [21] 进一步提高样本效率，尽管可能需要正则化来避免不稳定 [22]。至关重要的是，离策略方法还可以融入人类演示数据来引导学习（如 [23]），结合模仿学习和 RL 的优势。越来越多的研究已经开发了在物理机器人上部署 RL 的实用方案，包括自主数据收集流水线 [24]、高效学习框架（如 SERL [4, 25] 和 RL100 [5]），以及允许操作者在自主执行过程中介入并提供纠正的人机协作变体 [4]。这些系统已经证明，离策略 actor-critic 方法结合演示和人类纠正，可以在数小时机器人时间内解决接触丰富的操作任务。然而，它们通常从标准预训练视觉编码器（如 ResNet）之上从头训练小型策略，放弃了现代 VLA 模型中可用的丰富行为先验。RLT 弥合了这一差距，将冻结的 VLA 同时作为感知主干和行为先验用于轻量级在线 RL 策略。

**VLA 模型的 RL 微调。** 一个快速发展的研究方向是研究如何通过 RL 改进预训练的 VLA。这些方法的主要区别在于更新什么以及 RL 信号如何融入。在谱系的一端，若干方法更新完整的 VLA 模型。RECAP [3] 通过基于优势条件策略提取的离线 RL 端到端地训练整个 $\pi_{0.6}^{*}$ 模型：一个分布式价值函数估计每个时间步的优势，VLA 在所有收集的数据上进行训练——包括演示、自主 rollout 和人类干预——并使用最优性指示器对高优势动作进行加权。通过在机器人数据收集和离线 RL 更新之间迭代，RECAP 在复杂的长时间任务（如制作浓缩咖啡、折叠衣物和盒子组装）上实现了超过两倍的吞吐量提升。其他工作则应用近端策略优化（PPO）或其变体进行 VLA 微调（如 [1, 26, 27]），尽管同策略方法难以以样本高效且可扩展的方式扩展到真实世界 RL。在谱系的另一端，轻量级方法避免更新完整的 VLA，转而在冻结模型之上训练小型辅助模块。ConRFT [28] 冻结 VLA 编码器，使用基于一致性的训练目标结合学习的二值奖励分类器微调动作头，但仅对短视界任务的单步动作操作（无分块）。Policy Decorator [29] 学习一个残差策略，其输出由一个手动调节的超参数缩放后加到冻结 VLA 的预测上，但仅在仿真中以高样本需求（百万步量级）进行了演示。Probe-Learn-Distill (PLD) [30] 使用 Cal-QL [31] 在基础策略 rollout 上预训练 critic，然后在冻结 VLA 之上学习单步残差策略，可选择性通过监督微调将结果蒸馏回 VLA。GR-RL [2] 采用多阶段方法对长视界系鞋带任务上的通用 VLA 进行特化：首先执行离线过滤 BC，然后通过学习一个噪声预测器进行在线 RL，在潜空间中引导冻结 VLA 的扩散过程 [32]。DSRL [32] 同样在扩散噪声空间中操作，学习一个潜策略来调节去噪过程，将动作引导向高回报区域。

RLT 与这些方法共享通过 RL 改进预训练 VLA 而不需完整模型 RL 成本的目标，但在几个关键设计选择上有所不同。首先，RLT 引入了一个 RL token——一种训练用于压缩 VLA 内部嵌入的紧凑读出表示——作为轻量级 actor-critic 的状态观测，既保留了 VLA 的预训练感知结构，又能实现高效的在线学习。其次，RLT 在分块动作上操作，与 VLA 的原生动作接口对齐，缩短了在高控制频率下稀疏奖励的时序差分学习的有效决策视界——这与面临更长信用分配问题的单步方法 [28–30] 形成对比。第三，RLT actor 不预测残差或潜噪声，而是直接以 VLA 采样的参考动作块为条件并朝向其进行正则化，将在线 RL 转化为对良好 VLA 先验行为策略的局部精化，而非无约束搜索或扩散过程的隐式调制。这些选择的结合共同实现了真实机器人上的样本高效在线 RL——在几小时练习内同时提高成功率和执行速度。

# III. 预备知识

**视觉-语言-动作模型。** 大规模 VLA 模型从多样的人类演示数据集中学习操作行为，这些数据集涵盖数万小时，有时还补充了非机器人的视觉-语言数据 [7, 9, 16]。典型的 VLA 由两个组件组成：(i) VLM 主干，即一个视觉-语言模型，将多模态输入（图像、语言指令和本体感知状态）编码为共享的 token 序列；(ii) 动作专家，即一个基于扩散的模块，关注主干 token 并通过迭代去噪生成连续动作。我们基于 $\pi_{0.6}$ 模型 [33] 构建。给定最多四张相机图像、一个语言指令 $\ell$ 和本体感知状态 $\mathbf{s}_t^{\mathrm{p}}$，$\pi_{0.6}$ 生成一个动作序列（称为动作块）：$\tilde{\mathbf{a}}_{t : t + H - 1} = ( \tilde{\mathbf{a}}_{t} , \ldots , \tilde{\mathbf{a}}_{t + H - 1} ) \in \mathbb{R}^{H \times d}$，即对应于 1 秒控制的 $H = 50$ 个动作的序列。我们将预训练 VLA 产生的分块策略记为 $\pi_{\mathrm{vla}}$。在实践中，机器人在从新观测重新规划之前仅开环执行该块的前缀部分（如前 20 步）。由于某些任务的难度（如高精度任务），为其大规模收集高质量模仿学习数据可能具有挑战性，这限制了 VLA 在这些任务上的性能。这促使我们开发下一节中的在线 RL 精化方法。

**强化学习与 actor-critic 方法。** 我们将机器人控制形式化为马尔可夫决策过程（MDP）$( \mathcal{S}, \mathcal{A}, p, r, \gamma )$，其中 $s$ 是状态观测空间，$\mathcal{A}$ 是连续动作空间，$p( \mathbf{s}_{t+1} \mid \mathbf{s}_{t}, \mathbf{a}_{t} )$ 表示转移动力学，$r( \mathbf{s}_{t}, \mathbf{a}_{t} )$ 是奖励函数，$\gamma \in [0, 1)$ 是折扣因子。RL 的目标是学习一个最大化期望折扣回报的策略 $\pi( \mathbf{a}_{t} \mid \mathbf{s}_{t} )$：$\mathcal{J}(\pi) = \mathbb{E}_{\tau \sim \rho_{\pi}} \left[ \sum_{t=0}^{T} \gamma^{t} r_{t} \right]$，其中 $\rho_{\pi}(\tau)$ 表示策略 $\pi$ 诱导的轨迹分布。我们假设只能获得稀疏的二值奖励：人类监督者将每个 episode 的结束标记为成功或失败，我们设定成功时 $r_{T} = 1$，否则 $r_{T} = 0$。策略 $\pi$ 的动作-价值函数为 $Q^{\pi}( \mathbf{s}_{t}, \mathbf{a}_{t} ) = \mathbb{E}_{\tau \sim \rho_{\pi}} \left[ \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \mid \mathbf{s}_{t}, \mathbf{a}_{t} \right]$。

在我们的设定中，策略和 critic 都在动作块 $\mathbf{a}_{t : t + C - 1} = ( \mathbf{a}_{t}, \ldots, \mathbf{a}_{t + C - 1} ) \in \mathbb{R}^{C \times d}$ 上操作，其中 $C$ 表示 RL 块长度（而 $H$ 表示 VLA 预测的块视野）。我们选择 $C < H$ 以使策略更具反应性。我们将分块策略定义为 $\pi( \mathbf{a}_{t : t + C - 1} \mid \mathbf{s}_{t} )$ 以及对应的块级 C 步价值估计 $Q^{\pi}( \mathbf{s}_{t}, \mathbf{a}_{t : t + C - 1} ) = \sum_{t' = t}^{t + C - 1} \gamma^{t' - t} r_{t'} + \gamma^{C} \mathbb{E}_{\mathbf{a}' \sim \pi \mid \mathbf{s}_{t + C}} \left[ Q^{\pi}( \mathbf{s}_{t + C}, \mathbf{a}' ) \right]$。我们基于经典的离策略 actor-critic 方法 [17, 19, 34]，联合训练一个随机 actor $\pi_{\theta}$ 和一个 critic $Q_{\psi}$。至关重要的是，学习是离策略的，使用存储在回放缓冲区 $\mathcal{B}$ 中的转移样本，无论它们是由哪个策略生成的。这一特性在我们的设定中至关重要，因为 $\mathcal{B}$ 聚合了来自 VLA 策略、RL 学习器和人类遥操作干预的数据。

# IV. 基于 RL Token 的强化学习

图 1 总结了我们利用 RLT 实现从预训练 VLA 模型进行快速稳定在线 RL 的方案。核心思想是最大化利用预训练 VLA 来提高 RL 训练过程的效率。使用在线 RL 训练整个 VLA 可能计算和样本效率过低，无法在几小时内产生改进的策略。相反，我们使用冻结的 VLA 来提供 RL 状态表示、供应参考动作并引导探索朝向其自身预测附近的动作，同时仍然使用小型的 actor 和 critic 网络。我们首先在少量任务特定演示数据上适配 VLA，既改进其初始任务策略，又为下游 RL 暴露一个 RL token。然后我们冻结 VLA，在线训练轻量级的离策略 actor 和 critic 网络，同时以 RL token 表示和 VLA 的参考动作为条件，并正则化学习的策略使其保持接近 VLA 模型。我们的方法将在线 RL 转化为对有前景行为的局部精化，而非无约束搜索。这种设计使在线 RL 方法具有小型 actor-critic 算法的效率，同时保留预训练 VLA 模型的表示和行为。

## A. 适配 VLA 以暴露 RL 接口

样本高效的在线 RL 关键取决于状态表示的选择。直接将 RL 应用于完整的 VLA 模型不适合快速真实世界适应：表示是高维的，对数十亿参数模型的在线更新既计算昂贵又样本低效。同时，我们希望利用 VLA 预训练后已经包含的表示，因为它在大规模网络和机器人数据上训练，已经包含对许多任务生成动作有用的信息。然而，通常不明显基于 Transformer 的 VLA 中的哪些特征构成在线 RL 的良好表示，且每个 Transformer 层中的嵌入是高维的。因此，我们的目标是将 VLA 表示压缩到一个紧凑的嵌入中用于 RL，既保留任务相关信息，又保持足够小以用于轻量级在线 actor-critic 学习。

我们通过添加一个 **RL token**（图 2）来实现这一点：一个学习的读出嵌入，将 VLA 的知识总结为一个作为 RL 状态的小向量。具体来说，我们从一个添加到预训练 VLA 的小型额外 Transformer 中获取 RL token。我们以编码器-解码器 [35] 的方式训练该 Transformer，编码器的最后一个输入是 RL token。由于 RL token 的表示必须保留足够的信息以使解码器能够重建输入，它起到了信息瓶颈的作用。令 $\mathbf{z} = f( s, \ell ; \theta_{\mathrm{vla}} )$ 表示预训练 VLA 为状态 $s$ 和语言指令 $\ell$ 产生的最终层 token 嵌入。嵌入 $\mathbf{z}$ 分解为 $\mathbf{z}_{1:M} = \{ \mathbf{z}_{1}, \ldots, \mathbf{z}_{M} \}$，其中每个 $\mathbf{z}_{i}$ 对应一个输入 token 的嵌入。我们将一个学习的嵌入 $\mathbf{e}_{\mathrm{rl}} = \mathbf{e}_{\phi}( <\mathrm{rl}> )$ 附加到序列中，并用一个轻量级编码器 Transformer $g_{\phi}$ 处理增强后的序列。编码器在特殊 token 位置的输出，记作 $\mathbf{z}_{\mathrm{rl}}$，即我们的 RL token[^1]：

$$
\mathbf{z}_{\mathrm{rl}} = g_{\phi} \left(\left[ \mathbf{z}_{1: M}, \mathbf{e}_{\mathrm{rl}} \right]\right)_{M + 1}. \tag{1}
$$

[^1]: 在我们的实验中，每个任务都有固定的语言指令，因此我们在这一步中丢弃语言嵌入；该构造通常适用于所有 VLA 嵌入。

一个带有线性输出投影 $h_{\phi}$ 的解码器 Transformer $d_{\phi}$ 随后被训练以自回归地从 $\mathbf{z}_{\mathrm{rl}}$ 重建原始嵌入。令 $\bar{\mathbf{z}}_{i} = \mathrm{sg}( \mathbf{z}_{i} )$ 表示应用于 VLA 嵌入的停止梯度操作，则演示数据 $\mathcal{D}$ 上的自回归重建目标为：

$$
\mathcal{L}_{\mathrm{ro}} = \mathbb{E}_{\mathcal{D}} \left[ \sum_{i = 1}^{M} \left\| h_{\phi} \left(d_{\phi} \left(\left[ \mathbf{z}_{\mathrm{rl}}, \bar{\mathbf{z}}_{1: i - 1} \right]\right)\right)_{i} - \bar{\mathbf{z}}_{i} \right\|^{2} \right]. \tag{2}
$$

我们在少量任务特定演示数据集上训练参数 $\phi$，VLA 相对于 $\mathcal{L}_{\mathrm{ro}}$ 被视为冻结，（可选地）结合 VLA 的监督微调（$\theta_{\mathrm{vla}}$）。之后，$\theta_{\mathrm{vla}}$ 和 $\phi$ 都被冻结，在线 RL 在 RL token 表示 $\mathbf{z}_{\mathrm{rl}}$ 上操作。

![](images/648898e52dd511ea0f0f7c6b8705f99cd2b35a05d7e1b36b587d28a2d0318d87.jpg)
**图 2：** RL token 提取的细节。RLT 在预训练 VLA 上添加一个编码器-解码器 Transformer。它产生 VLA 表示的压缩嵌入（RL token）。该表示随后在在线 RL 期间实现数据和参数高效的微调。

## B. 在线 RL 精化 VLA 动作块

在初始适配阶段之后，我们冻结 VLA 和 RL token 表示。然后我们在线训练轻量级的 actor（$\pi_{\theta}$）和 critic（$Q_{\psi}$）网络。它们的输入 $\mathbf{x}$ 将 RL token 与任何有助于实现闭环控制的附加信息（如机器人的本体感知状态）组合在一起。Critic 模型估计状态和动作的价值：$Q_{\psi}( \mathbf{x}, \mathbf{a}_{1:C} ) \in \mathbb{R}$。值得注意的是，RL actor $\pi_{\theta}( \cdot \mid \mathbf{x}, \tilde{\mathbf{a}}_{1:C} )$ 并非从零开始生成动作，而是被训练来精化 VLA 提出的动作序列 $\tilde{\mathbf{a}}_{1:C}$（称为动作块）。

**训练 Critic。** 我们的 critic $Q_{\psi}( \mathbf{x}, \mathbf{a}_{1:C} )$ 以状态和动作块 $\mathbf{a}_{1:C}$ 作为输入。我们使用标准的离策略时序差分学习在从回放缓冲区 $\mathcal{B}$ 采样的动作块转移上训练 critic：

$$
\mathcal{L}_{Q} = \mathbb{E}_{(\mathbf{x}, \mathbf{a}_{1: C}, \mathbf{x}') \sim \mathcal{B}} \left[ \left(\hat{Q} - Q_{\psi} (\mathbf{x}, \mathbf{a}_{1: C})\right)^{2} \right],
$$

$$
\hat{Q} = \sum_{t' = 1}^{C} \gamma^{t' - 1} r_{t'} + \gamma^{C} \mathbb{E}_{\mathbf{a}' \sim \pi_{\theta}} \left[ Q_{\psi'} \left(\mathbf{x}', \mathbf{a}'\right) \right]. \tag{3}
$$

其中输入状态为 $\mathbf{x} = ( \mathbf{z}_{\mathrm{rl}}, \mathbf{s}^{\mathrm{p}} )$，$\mathbf{s}^{\mathrm{p}}$ 表示本体感知状态信息，$\mathbf{z}_{\mathrm{rl}}( \mathbf{s} )$ 表示为状态 $s$ 提取的 RL token；$\mathbf{x}'$ 表示下一个输入状态；$\mathbf{a}' \sim \pi_{\theta}$ 表示从 RL 策略中采样。在实践中，我们遵循 TD3 [19]，$\psi'$ 是目标网络的参数。

**训练 RL 策略。** 我们的 actor 网络 $\pi_{\theta}( \cdot \mid \mathbf{x}, \tilde{\mathbf{a}}_{1:C} )$ 产生动作块上的高斯动作分布。它以输入状态和参考动作块 $\tilde{\mathbf{a}}_{1:C}$ 为输入，产生动作分布：

$$
\pi_{\theta} \left(\mathbf{a}_{1: C} \mid \mathbf{x}, \tilde{\mathbf{a}}_{1: C}\right) = \mathcal{N} \left(\mu_{\theta} (\mathbf{x}, \tilde{\mathbf{a}}_{1: C}), \sigma^{2} \mathbf{I}\right), \tag{4}
$$

其中如前所述，$\mathbf{x} = ( \mathbf{z}_{\mathrm{rl}}, \mathbf{s}^{\mathrm{p}} )$。以 $\tilde{\mathbf{a}}$ 为条件直接将 actor 暴露于 VLA 预测的动作，使得在线 RL 精化一个强初始提案而非从头学习。第二个好处是采样的参考块保留了 VLA 多模态动作分布中的模式信息，这对于单峰高斯 actor 来说原本难以恢复 [36]。我们进一步通过将其动作正则化朝向参考动作来稳定学习。具体而言，我们优化 actor 以最大化 critic 价值同时保持在 VLA 参考块 $\tilde{\mathbf{a}}$ 附近，精神上类似于 KL 正则化 RL 方法（参见 [20, 37–40]）。这实际上将在线 RL 转化为围绕 VLA 生成的动作分布的局部动作编辑，而非在高维动作块上的无约束搜索。学习 RL 策略的目标为：

$$
\mathcal{L}_{\pi}(\theta) = \mathbb{E}_{\substack{\mathbf{s}\sim \mathcal{B}\\ \mathbf{a}_{1:C}\sim \pi_{\theta}}}\left[-Q_{\psi}(\mathbf{x},\mathbf{a}_{1:C}) + \beta \| \mathbf{a}_{1:C} - \tilde{\mathbf{a}}_{1:C}\|_{2}^{2}\right],
$$

$$
\tilde{\mathbf{a}}_{1: C} \sim \pi_{\mathrm{vla}} (\cdot \mid \mathbf{s}, \ell), \tag{5}
$$

其中系数 $\beta$ 控制 actor 朝向采样的 VLA 动作的正则化强度。

**参考动作 Dropout。** 参考动作条件化的一个实际失效模式是 actor 可能简单地复制 $\tilde{\mathbf{a}}$ 而不是学习改进它。这在 critic 变得有信息量之前尤其可能发生，因为以 $\tilde{\mathbf{a}}$ 为条件和朝向其正则化都鼓励 actor 保持接近 VLA 提案。为防止这一点，我们应用参考动作 dropout：对于每个训练批次中的随机子集转移，我们在将其传递给 actor 之前将参考块替换为零。这迫使 actor 保持独立的动作生成通路，同时在参考块存在时仍允许其利用 VLA 动作分布。在实践中，一旦 critic 提供有用的信号，actor 自然学会在偏离参考能增加预测价值时这样做。

# V. 完整系统

算法 1 总结了我们的完整训练循环。在初始预热阶段收集基础 VLA 策略的 episode 之后，训练交替进行在机器人上收集经验和从回放缓冲区进行离策略 actor-critic 更新。回放缓冲区聚合 VLA 预热数据、在线 RL rollout 和可选的人类干预。此外，人类监督者提供稀疏的成功/失败标签。步骤如下详述。

**预热。** 在训练 RL token 表示（第 IV-A 节）之后，我们通过展开 VLA 参考策略 $N_{\mathrm{warm}}$ 个环境步来预填充回放缓冲区 $\mathcal{B}$。这为 critic 提供了初始学习信号，并确保在线 RL 从有能力的 VLA 行为开始。

**Rollout。** 在在线收集期间的每个动作块边界，冻结的 VLA 产生参考块 $\tilde{\mathbf{a}}_{1:H}$，RL token 模块提取 $\mathbf{z}_{\mathrm{rl}}$。然后 actor 输出一个动作块 $\mathbf{a}_{1:C} \sim \pi_{\theta}( \cdot \mid \mathbf{x}, \tilde{\mathbf{a}}_{1:C} )$。为加速接触丰富或安全关键行为的学习，人类操作者可以选择性地介入，提供遥操作命令 $\mathbf{a}_{1:C}^{\mathrm{h}}$ 在干预期间覆盖 actor 输出。当发生这种情况时，干预替换回放缓冲区中的 VLA 参考。在所有情况下，存储在 $\mathcal{B}$ 中的每个转移包含执行的动作和对应的参考，使 actor 能够从自主 rollout 和人类纠正中学习。

**子采样动作块。** 虽然 RL 策略使用长度为 $C$ 的动作块，但我们获得每个中间步的观测。因此，我们可以通过将中间步存储到回放缓冲区来增加数据并提高学习效率。具体而言，我们选择步幅为 2，保存对应于 $< \mathbf{x}_{0}, \mathbf{a}_{0:C} >, < \mathbf{x}_{2}, \mathbf{a}_{2:C+2} >, < \mathbf{x}_{4}, \mathbf{a}_{4:C+4} >, \dots$ 的转移到回放缓冲区。注意，由于我们的 RL 算法的离策略特性，我们可以使用所有动作块（包括 VLA 生成的动作和人类干预）。

**更新。** 策略更新按照算法 1 从回放缓冲区离策略地执行。为在训练期间保持计算和时间效率，我们异步地执行 rollout 和学习。在实践中，我们为每个 actor 更新执行两次 critic 更新，并在预热阶段后不久开始学习。我们使用高达 5 的更新-数据比，这在低数据在线场景中至关重要。

```
算法 1: RLT

Require: 冻结的 VLA 主干 f_{θ_vla} 和 VLA 动作分布 π_vla; 演示数据 D，块长度 C，回放缓冲区 B，
         预热步数 N_warm，比率 G，VLA 微调权重 α，策略约束 β。

1: 训练 RL token 并（可选地）微调 VLA
2: 使用 z_i = f_i(s, ℓ; θ_vla)，z_rl = g_φ([z_{1:M}, e_rl])_{M+1} 训练 φ，
   以及 θ_vla（仅当 α > 0 时）
   L_ro(φ) = E_D [Σ_{i=1}^{M} || h_φ(d_φ([z_rl, z̄_{1:i-1}]))_i - z̄_i ||^2]
3: φ, θ_vla = arg min_{φ, θ_vla} L_ro(φ) + α L_vla(θ_vla)
4: 训练 RL actor 和 critic
5: 初始化 critic Q_ψ 和 RL 策略 π_θ
6: for 环境步 t = 0, C, 2C, ... do
7:     采样 VLA 参考块 ã_{t:t+C-1} ~ π_vla(s_t)
8:     形成 RL 状态 x_t = (z_rl(s_t), s_t^p)
9:     选择动作：
       a_{t:t+C-1} ← { a^human   (如果干预)
                      { ã_{t:t+C-1} (如果 t < N_warm)
                      { ~ π_θ(· | x_t, ã) (其他情况)
10:    执行 a_{t:t+C-1} 并观测 r_t, s_{t+1}, s_{t+1}^p
11:    如果干预，则 ã_{t:t+C-1} ← a^human
12:    存储转移到 B：x_t, a_{t:t+C-1}, ã, r_t, x_{t+1}
13:    for g = 1, ..., G do
14:        从 B 采样批次数据 b
15:        计算目标 Q 值：
          Q̂ = Σ_{t'=1}^{C} γ^{t'-1} r_{t'} + γ^{C} E_{a'~π_θ}[Q_{ψ'}(x', a')]
16:        使用 TD 备份训练 Critic（公式 (3)）：
          L_Q(ψ) = E_b[(Q̂ - Q_ψ(x, a))^2]
17:        训练策略 a ~ π_θ(· | s, ã)（公式 (5)）：
          L_π(θ) = E_b[-Q_ψ(x, a) + β ||a - ã||_2^2]
18:    end for
19: end for
```

**针对关键阶段的有针对性改进。** 为了学习的实用性和效率，我们将 RLT 应用于改进每个任务的关键阶段——对应于需要高精度的最困难部分——而让基础 VLA 执行任务中较容易的部分。具体而言，每个 episode 从执行基础模型开始。在数据收集期间，人类操作者可以选择在哪个时刻将控制权从基础 VLA 移交给 RL 策略。这类似于交互式模仿学习 [41] 中的人类干预决策。然后，我们的系统将 RL 应用于选定的任务段，在该关键阶段存储和训练转移，直到收到来自人类操作者的终端信号指示 RL 任务的成功或失败。这将数据收集和信用分配集中在在线适应最重要的行为部分。为了在测试时实现自主执行，我们可以通过要求 VLA 额外预测何时移交给 RL 策略（使用人类干预作为标签）来以最后的短微调阶段结束训练。然后，我们可以在测试时自动触发策略切换。

# VI. 真实世界实验

我们在四个需要灵巧控制和亚毫米精度的真实世界操作任务上评估 RLT。预训练的 VLA 为这些任务的大部分环节提供了强大的初始化，但成功和速度最终取决于精化需要最高精度的接触丰富关键阶段。我们的实验测试了我们的方法是否能在激励该方法的实际约束下实现此类改进：有限的机器人交互时间、稀疏的人类监督和轻量级的在线学习。

我们围绕以下问题组织评估：

**Q1.** RLT 能否在基础 VLA 模型之上改进操作性能？
**Q2.** RLT 在这些任务上与其他 RL 方法相比如何？
**Q3.** 方法的每个组件——RL token、分块动作预测、策略正则化和参考动作直通——对方法性能的贡献有多大？
**Q4.** RLT 是否使策略能够发现更好的策略，以及其策略与原始演示数据相比如何？

## A. 任务与设置

我们在以下任务上评估我们的方法（图 3）：

- **螺丝安装。** 机器人必须使用电动螺丝刀将 M3 螺丝拧入螺纹插座。这需要螺丝头和螺丝刀尖之间的亚毫米对准。该任务特别困难，因为 (1) 螺丝可能不总是完美竖直；(2) 握住螺丝刀时，末端执行器的任何旋转都会被螺丝刀尖与抓取点之间 10 cm 的距离放大；(3) 关键的视觉线索主要从对侧手臂的广角腕部相机可见，呈现出具有挑战性的感知问题。
- **扎带紧固。** 机器人必须将扎带尾部穿过其狭窄的锁定槽。该任务涉及对可变形物体的协调双臂控制，公差严格。成功插入需要仅从腕部相机推断尖端和槽的位置，并以毫米级精度执行。
- **以太网插入。** 机器人必须将以太网接头插入凹入式端口。这需要精确的位置和角度对准，然后进行果断有力的插入动作。小的方向误差或犹豫的接触通常导致接头卡在壳体上而非插入端口，使成功对精度和接触动力学都敏感。
- **充电器插入。** 机器人必须对准并将充电器插入电源插座。任务困难之处在于策略必须实现厘米级对准，同时对插脚和插座并非始终具有清晰的可观测性。小的对准误差通常导致反复试探或插入尝试失败。

每个任务包括抓取、重新定位和对准，持续 30–120 秒（在 $50\mathrm{Hz}$ 下约为 1500–6000 个控制步）。对于每个任务，我们确定关键阶段——插入、紧固或旋转段——其中精度要求最高且基础 VLA 最常减速或失败。这些阶段通常持续 5–20 秒（250–1000 个控制步）。

**关键阶段评估。** 由于我们的方法旨在精确改进这些关键阶段，我们首先将评估重点放在仅在关键阶段比较方法和消融实验上。在此设置中，episode 在部分完成的任务状态被重置到关键阶段之前开始，使用略微随机化的初始配置集合。例如，在扎带紧固中，机器人在插入尝试开始之前已经开始握住扎带的两端。此设置隔离了 RL 预期最重要的精度关键段，并减少了任务早期阶段（如抓取和运输）的混淆方差，这些阶段基础 VLA 已经处理得相当好。每个智能体在此受控设置下每个任务评估 50 个 episode。

**完整任务评估。** 受控的关键阶段评估有助于隔离我们方法旨在改进的瓶颈，但不能捕捉长视界执行的完整变异性。因此，我们额外在更真实的设置中评估完整任务性能，其中机器人从其"起始位置"开始，用基础策略执行任务的早期阶段，并在该执行诱导的状态变化下进入关键阶段。此设置难度显著更大，因为 RL 改进的行为必须在前置策略产生的更广泛状态分布下保持有效。对于完整任务训练，我们首先让 RL 专注于具有小随机化的关键阶段，然后进入完整任务设置。

**实验细节。** RL 策略输入包括 RL token（由两张腕部相机图像和一张基础相机图像产生）以及额外的本体感知状态。根据任务不同，此辅助状态可能包括关节位置（螺丝）、末端执行器姿态（扎带、以太网和充电器）。我们使用 $\pi_{0.6}$ [33] 作为基础 VLA 策略。机器人以 $50\mathrm{Hz}$ 的控制频率运行。对于 14 维的每时间步动作空间，这对应于 RL actor 的 140 维分块动作。我们在附录 B 中提供了更多实现细节。

## B. 基线方法与消融实验

我们从预训练的 VLA 模型 $\pi_{0.6}$ [33] 开始。对于每个任务，我们收集 1–10 小时的遥操作演示。然后我们在训练 RL token 表示的同时微调 VLA 模型。这产生了我们在所有实验中贯穿使用的基础 VLA 策略。我们根据任务难度运行 400 到 1000 个 episode 的 RL 训练。排除重置和各种开销，每个实验产生约 15 分钟到 5 小时的实际机器人数据。我们以每个任务的成功率来衡量性能，由人类操作者的二值奖励信号判断。我们还报告吞吐量（每 10 分钟间隔内成功完成任务的数量），以评估鲁棒性和速度两个维度的改进。我们在所有任务的关键阶段上进行评估，并对两个更难的任务——螺丝和扎带任务——在完整任务设置中进行评估。

我们将 RLT 与四种从经验中改进策略的基线方法进行比较。为公平比较，我们为每种 RL 方法训练相同数量的数据（参见附录 C）。

- **HIL-SERL [4]：** 与我们的方法类似，HIL-SERL 结合经验和干预训练小型 actor 和 critic，但不同于 RLT，它不使用预训练 VLA 的表示，而是使用为标准计算机视觉任务预训练的简单 ResNet 编码器。
- **Probe-Learn-Distill [30]：** PLD 学习一个残差策略，为每个单步动作输出一个残差。它将残差乘以一个超参数缩放后与冻结 VLA 的动作预测中的一步相加以执行。
- **DSRL [32]：** DSRL 在流式 VLA 模型的潜噪声空间中学习在线 RL 策略。它通过选择馈入冻结 VLA 模型动作生成器的噪声来"引导"VLA 动作生成。该方法隐式地将探索约束在 VLA 能够生成的那些动作上，并在其模式之间探索。
- **DAgger [41, 42]：** 我们在训练期间收集的人类干预数据上微调基础 VLA 模型。

我们还通过逐个移除来隔离我们方法每个组件的贡献：

- **w/o RL token：** 将 RL token 替换为来自 [25] 的冻结、ImageNet 预训练 ResNet-10 编码器。
- **w/o Chunk：** RL 策略输出单步动作（$C = 1$）而非动作块。因为该策略需要在 $50\mathrm{Hz}$ 下运行，而查询基础 VLA 模型在 $50\mathrm{Hz}$ 下不可行，我们必须将 RL token 替换为 ResNet-10 编码器。
- **w/o BC Regularizer：** 在公式 (5) 中设 $\beta = 0$；策略仅使用 $Q$ 函数训练。
- **w/o Pass-Through：** 从公式 (4) 的策略输入中移除 $\tilde{\mathbf{a}}$；RL actor 仅从状态和 RL token 生成动作。

## C. 实验结果

**Q1：在线 RL 是否在基础 VLA 策略之上有所改进？** 我们在两种场景下评估我们的方法：隔离关键阶段的受控设置和要求 RL 策略更具鲁棒性的完整任务设置。在线 RL 在两种设置下都改进了基础模型的成功率和执行速度。在受控设置中，RLT 一致地改进了所有四个任务的关键阶段。即使在基础策略已经实现良好可靠性的相对简单的充电器和以太网任务上，RLT 学到的策略在关键阶段也快了约 $3 \times$。在更难的扎带和螺丝刀任务上，成功率的提升更加明显。在完整任务评估中，由于任务早期部分（抓取/举起物体等）的复合误差，总体成功率较低，但 RLT 仍然将螺丝刀任务的成功率提高了 $40\%$，将扎带任务的成功率提高了 $60\%$。

**Q2：RLT 与其他方法相比如何？** 如图 6 所示，与基线方法相比，RLT 显著提升了吞吐量。我们在以太网任务上比较了四种基线。HIL-SERL 和 PLD——两者都是单步在线 RL 方法——未能在此任务上有效学习，该任务跨越数百步且具有稀疏奖励。没有动作分块，任务的视界非常长，价值函数更新无法有效传播稀疏奖励信号。对于这个较简单的任务，DAgger 和 DSRL 达到了与 RLT 相当的成功率（图 6），但在速度方面提供的改进明显较少。DAgger 是一种模仿学习方法，受限于人类演示和干预的速度。DSRL 是一种 RL 方法，强烈约束策略保持接近基础 VLA，提供稳定的训练但改进潜力相对较小。相比之下，RLT 匹配了基础策略的高成功率，同时将完成平均步数减少了 $2 \times$。

**Q3：每个组件的贡献有多大？** 所有四个设计选择——RL token、动作分块、BC 正则化器和参考动作直通——都有实质性的贡献（图 7）。我们验证了我们方案的每个组件都提供了正向贡献：将 RL token 替换为 ResNet-10 编码器将吞吐量减少了 $50\%$，证实了我们的 token 编码了标准计算机视觉任务训练的现成编码器所不能提供的操作相关结构。将分块（$C=10$）替换为单步动作显著增加了任务的有效视界，因为价值函数需要在长得多的视界上执行信用分配。这也使得使用 RL token 运行我们的方法不可行。在实践中，单步变体无法可靠地匹配基础策略性能。移除 BC 正则化器（$\beta = 0$）导致最大的单一性能下降，因为它迫使 actor 仅凭借来自 $Q$ 函数的梯度探索整个动作空间。移除参考动作直通会减慢学习速度，导致早期探索漂移，并偶尔出现退化行为。对于这个较简单的任务，该消融最终确实匹配了 RLT 的性能，但在训练过程中经历了更多的失败，如图 7 的学习曲线所示。

**Q4：RLT 是否带来了更有效的新兴策略？** 超越聚合指标，在线 RL 的效果体现在机器人执行任务方式的质变上。在以太网任务的关键阶段，我们可视化了遥操作演示、基础策略和最终 RL 策略的速度分布（图 9）。基础 VLA 在接触附近经常表现出"试探"行为：它接近目标，略微后退，重新调整，然后再次尝试——有时在成功之前循环几次这样的尝试。RLT 则直接接近端口，以流畅的动作插入接头。即使在第一次尝试失败时，RLT 也会施加压力并轻轻摆动接头以利用顺应性，从而实现更快的插入。这种行为未见于演示数据中，纯粹从在线探索中出现，说明该方法可以超越模仿人类策略。

# VII. 结论

我们提出了 RLT，一种在从大型预训练 VLA 提取的表示之上进行快速在线 RL 的方法。通过训练 VLA 暴露一个紧凑的表示，我们的方法使轻量级的 actor 和 critic 能够仅用几小时的真实世界练习来改进高精度和精细的任务。在四个需要精度和速度的困难任务中，RLT 一致地改进了成功率和执行速度，将每个任务最难阶段的速度提升高达 $3 \times$，在某些情况下通过在线 RL 涌现的策略甚至超越了专家人类遥操作速度。

虽然 RLT 提供了快速高效的学习，但它确实在训练期间需要额外的人类干预来提供奖励信号、干预纠正以及在 RL（用于关键阶段）和基础策略（用于其他阶段）之间切换。原则上，这些组件中的一些可以被自动化，例如通过使用奖励模型和进度预测。基于 RLT 开发完全自主的 RL 改进流水线是未来工作的一个有前景的方向。更广泛地说，我们相信我们的方法代表了向不仅能从演示数据学习、而且能直接在任务中改进的机器人系统迈出的重要一步。当改进快速且可靠时，VLA 的预训练阶段只需要为下游探索提供一个良好的初始化，而最成功和最优性能的策略可以通过强化学习来发现。我们希望 RLT 能够成为迈向这一未来的一步。

# 致谢

机器人技术是团队努力的结果。我们感谢 Physical Intelligence 所有为这项工作的各个方面做出贡献的人，包括硬件、数据收集、机器人操作和机器人基础设施。我们感谢 Liam Murphy 和 Cameron Myers 在夹爪设计上的帮助。我们感谢 PI 的机器人操作员以及操作和标注团队。我们感谢 Connor Jacobsen 在网站和博客文章上的帮助，Brian Ichter 在图表上的帮助，Kyle Vedder 在校对上帮助，Claudio Guglieri 在博客文章可视化上的帮助，以及 Donald Jewkes 和 Thomas Burton 在视频拍摄和编辑上的帮助。

# 参考文献

[1] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint, arXiv:2509.09674, 2025. 1, 2
[2] Yunfei Li, Xiao Ma, Jiafeng Xu, Yu Cui, Zhongren Cui, Zhigang Han, Liqun Huang, Tao Kong, Yuxiao Liu, Hao Niu, Wanli Peng, Jingchao Qiao, Zeyu Ren, Haixin Shi, Zhi Su, Jiawen Tian, Yuyang Xiao, Shenyu Zhang, Liwei Zheng, Hang Li, and Yonghui Wu. Gr-rl: Going dexterous and precise for long-horizon robotic manipulation, 2025. URL https://arxiv.org/abs/2512.01801. 2
[3] Physical Intelligence. $\pi_{0.6}^{*}$: a VLA That Learns From Experience, 2025. URL https://arxiv.org/abs/2511.14759. 1, 2
[4] Jianlan Luo, Charles Xu, Jeffrey Wu, and Sergey Levine. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. arXiv preprint arXiv:2410.21845, 2024. 1, 2, 7
[5] Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, and Huazhe Xu. Rl-100: Performant robotic manipulation with real-world reinforcement learning, 2026. URL https://arxiv.org/abs/2510.14830. 1, 2
[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In arXiv preprint arXiv:2307.15818, 2023. 2
[7] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2, 3
[8] Physical Intelligence. $\pi_{0}$: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 2
[9] Gemini Robotics Team, et al. Gemini robotics: Bringing ai into the physical world, 2025. URL https://arxiv.org/abs/2503.20020. 2, 3
[10] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pretraining for visual robot manipulation, 2023.
[11] NVIDIA, et al. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734. 2
[12] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL https://arxiv.org/abs/2304.13705. 2
[13] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 2
[14] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025. 2
[15] Suneel Belkhale and Dorsa Sadigh. Minivla: A better vla with a smaller footprint, 2024. URL https://github.com/Stanford-ILIAD/openvla-mini. 2
[16] Physical Intelligence. $\pi_{0.5}$: a vision-language-action model with open-world generalization. In 9th Annual Conference on Robot Learning, 2025. 2, 3
[17] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. Pmlr, 2018. 2, 3
[18] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.
[19] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018. 3, 4, 12
[20] Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Riedmiller. Maximum a Posteriori Policy Optimisation. In International Conference on Learning Representations (ICLR), 2018. 2, 4
[21] Marcel Hussing, Claas Voelcker, Igor Gilitschenski, Amir-massoud Farahmand, and Eric Eaton. Dissecting deep rl with high update ratios: Combatting value divergence, 2024. URL https://arxiv.org/abs/2403.05996. 2
[22] Xinyue Chen, Che Wang, Zijian Zhou, and Keith Ross. Randomized ensembled double q-learning: Learning fast without a model. arXiv preprint arXiv:2101.05982, 2021. 2
[23] Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, pages 1577–1594. PMLR, 2023. 2
[24] Henry Zhu, Justin Yu, Abhishek Gupta, Dhruv Shah, Kristian Hartikainen, Avi Singh, Vikash Kumar, and Sergey Levine. The ingredients of real-world robotic reinforcement learning. arXiv preprint arXiv:2004.12570, 2020. 2
[25] Jianlan Luo, Zheyuan Hu, Charles Xu, You Liang Tan, Jacob Berg, Archit Sharma, Stefan Schaal, Chelsea Finn, Abhishek Gupta, and Sergey Levine. Serl: A software suite for sample-efficient robotic reinforcement learning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 16961–16969. IEEE, 2024. 2, 7
[26] Allen Z. Ren, Justin Lidard, Lars Lien Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion Policy Policy Optimization. In Proceedings of the 2025 International Conference on Learning Representations (ICLR), 2025. 2
[27] Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, Tiejun Huang, Yu Wang, and Chao Yu. $\pi_{\mathrm{RL}}$: Online rl fine-tuning for flow-based vision-language-action models. arXiv preprint, arXiv:2510.25889, 2025. 2
[28] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025. 2, 3
[29] Xiu Yuan, Tongzhou Mu, Stone Tao, Yunhao Fang, Mengke Zhang, and Hao Su. Policy decorator: Model-agnostic online refinement for large policy model. In The Thirteenth International Conference on Learning Representations, 2025. 2
[30] Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi "Jim" Fan, Guanya Shi, and Yuke Zhu. Self-improving vision-language-action models with data generation via residual rl, 2025. 2, 3, 7
[31] Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. Advances in Neural Information Processing Systems, 36:62244–62269, 2023. 2, 12
[32] Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering your diffusion policy with latent space reinforcement learning. In Proceedings of the 9th Conference on Robot Learning (CoRL), 2025. 2, 7
[33] Physical Intelligence. $\pi_{0.6}$ model card, 2025. URL https://website.pi-asset.com/pi06star/PI06_model_card.pdf. 3, 7
[34] Nicolas Heess, Gregory Wayne, David Silver, Timothy Lillicrap, Tom Erez, and Yuval Tassa. Learning continuous control policies by stochastic value gradients. In Advances in Neural Information Processing Systems, volume 28, 2015. 3
[35] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pages 3104–3112, 2014. 4
[36] Seohong Park, Qiyang Li, and Sergey Levine. Flow q-learning. In International Conference on Machine Learning (ICML), 2025. 4
[37] Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang-Wei Lee, Jie Tan, and Sergey Levine. Learning agile robotic locomotion skills by imitating animals. RSS, 2020. 4
[38] Jan Peters, Katharina Mulling, and Yasemin Altun. Relative entropy policy search. In Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, AAAI'10, page 1607–1612, 2010.
[39] Peter Dayan and Geoffrey E. Hinton. Using expectation-maximization for reinforcement learning. Neural Computation, 9(2):271–278, 1997.
[40] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review, 2018. URL https://arxiv.org/abs/1805.00909. 4
[41] Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, and Mykel J. Kochenderfer. Hg-dagger: Interactive imitation learning with human experts, 2019. URL https://arxiv.org/abs/1810.02890. 6, 7
[42] Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15, pages 627–635, 2011. 7

# 附录

## A. 贡献

CX 和 LK 启动了该项目。CX 构建了在线 RL 的基础设施。JTS 设计并训练了 RL token。ME 构建了干预接口。AA 和 AE 设计并制造了夹爪和机器人硬件。CX 和 LK 设计了系统实现、任务套件和实验。SL 和 LK 在整个项目中提供了建议。LK、CX、JTS、SL 和 ME 参与了写作、插图和视频制作。

## B. 额外实验细节

首先，我们在目标任务上收集演示数据集；然后在单任务数据上微调基础 VLA 模型并训练 RL token 进行 2000 到 10000 个梯度步。在在线 RL 训练期间，VLA 被冻结。

在在线 RL 期间，我们对扎带紧固、以太网和充电器插入任务使用两层 MLP（隐藏维度 256）从头初始化 RL actor 和 critic。对更具挑战性的螺丝安装任务，我们使用更大的网络，由三层 MLP 组成，隐藏维度为 512。两个网络都以冻结基础 VLA 模型产生的 RL token、本体感知位置和速度作为输入。Critic 按照 Fujimoto 等人 [19] 的方法使用两个 Q 函数的集成进行训练，我们取两个 Q 函数的最小值来计算目标值。Actor 额外接收 VLA 模型产生的参考动作块，在训练期间以 $50\%$ 的概率被掩码，在推理期间始终提供。Actor 被参数化为具有小固定标准差的高斯策略，从当前观测输出动作块 $\mathbf{a}_{t:t+C-1} \in \mathbb{R}^{C \times d}$，其中 $C=10$。为提高样本效率，我们在训练期间以间隔 2 个控制步子采样动作块，因此每秒数据大约为 RL 网络产生 25 个样本。操作者在训练期间当 RL 任务完成时提供稀疏的 $+1$ 奖励。

对于螺丝安装和扎带紧固任务，我们首先仅在关键阶段设置中开始 RL 训练。然后我们进入完整任务阶段，首先运行基础模型完成任务中非关键的部分，在到达关键阶段时切换到 RL 策略。这种两阶段训练策略提高了训练效率，同时确保 RL 策略对基础策略在任务早期部分诱导的初始分布具有鲁棒性。我们报告收集约 5 小时数据后的策略性能。

## C. 基线方法的额外实验细节

对于所有基线方法，我们使用与我们的方法相同的环境和动作空间设置——策略在 $50\mathrm{Hz}$ 下的增量动作空间中执行。

**PLD：** 遵循原始论文，我们首先在 50 个基础策略 rollout 上使用 Cal-QL [31] 预训练 critic 网络以获得更好的样本效率。然后进入在线 RL 阶段。

**DSRL：** 遵循原始实现，我们的实现预测一个 $(1, 32)$ 维的潜动作，该动作在第一维上重复 50 次以匹配我们动作分块 VLA 的噪声输入空间。

**HIL-SERL：** 遵循原始实现，我们用 20 个 episode 的演示初始化 RLPD 训练，并在整个训练过程中提供干预。然而，由于与原始系统（$10\mathrm{Hz}$）相比更高的控制频率（$50\mathrm{Hz}$）以及没有动作空间边界框来减少探索空间，它无法在我们的设置中取得成功。

**DAgger：** 我们使用演示数据和在线 RL 训练期间收集的同一组干预数据的混合来微调我们的 VLA。
