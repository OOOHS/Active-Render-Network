# 开发日志 (Development Log)

## 2025-03-15：SAC → DDPG 重构（取消主动方差 + 衰减 OU 探索）

### 动机
- **取消主动方差输出**：原先 Actor 输出 (μ, log_σ) 的 SAC 风格在训练后期容易产生“手抖”（策略方差导致动作抖动），影响收敛与画质稳定性。
- **探索交给衰减的 OU 噪声**：希望训练前期有充分探索，后期逐渐退化为纯确定性策略，由外部噪声控制探索而非网络输出方差。

### 改动摘要

1. **Actor 网络（`modules/actor.py`）**
   - 移除方差相关：删除 `logstd_head`、`use_std_head`、`log_std_min/max` 及 `_get_mu_logstd` 中的 log_std 分支。
   - 仅保留确定性输出：`_get_action(I_star, C, t_emb)` → backbone → `mu_head` → `tanh(mu)`。
   - 删除 `sample()` 及 log π(a|s) 计算，保留 `forward` / `act_deterministic` 一致接口。

2. **探索策略（`systems/painter.py`）**
   - **Rollout**：warmup 之后不再调用 `actor.sample()`，改为 `a = act_deterministic(...) + scale * OU_noise`，再 `clamp(-1, 1)`。
   - **衰减**：使用 `LinearSchedule(ou_scale_init → ou_scale_end, duration=explore_steps)` 对 OU 噪声缩放系数做线性衰减，前期有探索，后期趋于 0，变为纯确定性策略。

3. **Actor 更新（DDPG）**
   - `_step_actor` 中不再使用 `actor.sample()` 与熵项 `entropy_alpha * logp_hat`。
   - 使用确定性动作 `a_hat = actor(I_s, C_s)`，目标为最大化 `r_b_total + gamma * q_val`，即标准 DDPG 的 Q 最大化。

4. **配置（`configs/default.yaml`）**
   - 移除：`actor_std_head`、`actor_log_std_min`、`actor_log_std_max`、`entropy_alpha`。
   - 新增：`explore_steps`、`ou_scale_init`、`ou_scale_end`、`ou_theta`、`ou_sigma`，用于衰减 OU 探索。

### 预期效果
- 后期“手抖”减轻：策略为确定性，抖动仅来自已衰减的 OU，随步数增加趋近于 0。
- 探索可调：通过 `explore_steps` 与 `ou_scale_*` 控制探索强度与衰减速度，便于调参。

## 2026-03-19：Renderer 残差先验 + 奖励/回合重标定

### 动机
- 训练不够稳定、重建效果偏弱：需要强化“每一步更新”的信用分配信号，并给 Renderer 一个更合理的先验。
- 保持你的理念：Active Render Network 作为连续序列解码器，需要从粗到细逐步逼近目标。

### 改动摘要
1. **Renderer 残差先验（`modules/renderer.py`）**
   - 由于 Renderer 的输出 head 做了零初始化，原实现等价于“从零重建 canvas”，导致前期极不稳定。
   - 改为残差更新：`C_next = C + delta`，在 head 仍接近 0 时会保持画布基本不变，从而稳定早期学习。

2. **配置重标定（`configs/default.yaml`）**
   - `horizon` 调整为 `64`：与 `num_patches=64` 对齐，支撑变长连续序列的信用分配链更合理。
   - `stop_tau` 提高到 `0.999`：强调高保真重建；即使 done 不频繁触发，依然会通过缓冲中的 `t>0.95` 写入 terminal 样本。
   - `shaping_scale` 提升到 `50.0`：让步级 shaping/MSE 增量形成可观测 reward 量级，增强 Q 学习信号。
   - 关闭 `msssim_lambda`：保持 `0.0`（你的要求）。
   - 提高 `d_steps` 到 `3`：让 WGAN-GP 的 discriminator/critic 对 actor/renderer 的更新保持足够领先。

3. **训练稳定性增强（`systems/painter.py`）**
   - GAN bonus：对 WGAN critic 输出做 `clamp(-10, 10)`，避免 reward/Q 目标无界导致 critic 发散、actor 抖动。
   - OU 噪声探索：在每个 episode 结束后调用 `ou.reset()`，避免跨 episode 的噪声状态漂移。

### 预期效果
- 前期保持画布“近似不变”的残差先验，使 Renderer 从粗到细的学习更平滑。
- reward 信号量级更合理（`shaping_scale`/`stop_tau`/`horizon` 配套），提升 Q 学习的稳定性与收敛速度。

## 2026-03-22：回退 SAC + 95 平台触发熵退火 + Renderer 对比一致性

### 动机
- 在保持当前稳定性改动（残差先验、reward 量级重标定等）的前提下，恢复 SAC 的随机策略表达与熵正则，降低确定性策略过早收敛的风险。
- 熵项不希望从一开始就快速衰减，而是希望在训练相似度“卡在 0.95 附近”时再退火，作为平台期后的精修开关。
- 为 Renderer 增加“输入-输出几何一致性”先验：输入状态/动作相近时输出应相近，输入差异大时输出也应拉开，减少不稳定映射。

### 改动摘要
1. **SAC 训练逻辑恢复（`modules/actor.py` + `systems/painter.py`）**
   - Actor 恢复 `μ + logσ` 分支与 `sample()`（含 tanh-squash 后的 `log π` 修正）。
   - Rollout 在 warmup 后恢复使用 `actor.sample(...)` 采样动作；移除 DDPG 阶段引入的 OU 衰减探索逻辑。
   - `_step_actor` 恢复 SAC 目标：`r + γ·Q - α·logπ`，并通过配置项读取熵系数。

2. **相似度平台触发熵退火（`systems/painter.py`）**
   - 新增按 rollout 末状态统计的平台检测：当 batch 平均相似度连续落在 `entropy_plateau_sim ± entropy_plateau_band`（默认 `0.95 ± 0.02`）达到 `entropy_plateau_patience` 后，记录退火起点。
   - 新增 `_current_entropy_alpha()`：从触发点开始，在 `entropy_anneal_duration` 内把 `entropy_alpha` 线性退火到 `entropy_alpha_end`。
   - 状态通过 buffer 挂在模块内（起始 step、连续计数），并日志化 `train/entropy_alpha` 便于观测。

3. **Renderer 对比一致性约束（`systems/painter.py`）**
   - 新增 `loss/renderer_consistency`：构建 batch 内输入特征（`C_s`、`z_hat`、`t_s`）与输出特征（`C_next_hat`）的两两距离矩阵，并做 SmoothL1 对齐。
   - 该损失以 `renderer_consistency_lambda` 加权并入 actor/renderer 总损失，实现“输入近/远 -> 输出近/远”的对比式约束。
   - 默认使用保守强度，避免压制主任务奖励信号。

4. **配置更新（`configs/default.yaml`）**
   - 熵退火：`entropy_alpha_end`、`entropy_anneal_duration`、`entropy_plateau_sim`、`entropy_plateau_band`、`entropy_plateau_patience`。
   - Renderer 一致性：`renderer_consistency_lambda`（默认 `0.05`）、`renderer_consistency_pool`（默认 `4`）。

### 预期效果
- 平台期前保持探索能力，平台期后自动降熵细化重建，减轻“早收敛但细节不足”。
- Renderer 映射更平滑、局部扰动更可控，降低输入微扰导致输出跳变的风险。
- 在不破坏现有稳定性增强策略的前提下，提升后期收敛质量与可调参性。
