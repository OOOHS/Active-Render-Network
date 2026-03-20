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
