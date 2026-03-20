# systems/painter.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.utils as vutils
from pytorch_msssim import ms_ssim
from dataclasses import asdict

from modules.actor import Actor
from modules.critic import Critic
from modules.vq import VectorQuantizer, IdentityVQ
from modules.renderer import Renderer
from modules.reward_discriminator import Discriminator, wgan_gp_loss
from modules import target_nets
from rl.utils import OUNoise, LinearSchedule
from rl.buffer import ReplayBuffer
from utils.metrics import mse_similarity


def _set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


class PainterSystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(asdict(cfg))

        # ---- nets ----
        self.actor = Actor(self.cfg.model)
        self.critic1 = Critic(self.cfg.model)
        self.critic2 = Critic(self.cfg.model)

        use_vq = bool(getattr(self.cfg.train, "use_vq", False))
        if use_vq:
            self.vq = VectorQuantizer(self.cfg.model.codebook_size, self.cfg.model.token_dim)
        else:
            self.vq = IdentityVQ()

        self.renderer = Renderer(self.cfg.model)

        # In: [Canvas, Target] -> 2*C
        self.D = Discriminator(in_ch=2 * self.cfg.model.canvas_channels)

        # ---- targets ----
        self.actor_t = target_nets.make_target(self.actor)
        self.critic1_t = target_nets.make_target(self.critic1)
        self.critic2_t = target_nets.make_target(self.critic2)
        self.D_t = target_nets.make_target(self.D)

        # ---- buffer & utils ----
        self.buf = ReplayBuffer(int(self.cfg.train.buffer_size))
        self.ou = OUNoise(
            action_dim=self.cfg.model.token_dim,
            theta=float(getattr(self.cfg.train, "ou_theta", 0.15)),
            sigma=float(getattr(self.cfg.train, "ou_sigma", 0.2)),
        )
        # 衰减的 OU 探索：前期有探索，后期趋于纯确定性（DDPG）
        explore_steps = int(getattr(self.cfg.train, "explore_steps", 100000))
        ou_scale_end = float(getattr(self.cfg.train, "ou_scale_end", 0.0))
        self.ou_scale_schedule = LinearSchedule(
            start=float(getattr(self.cfg.train, "ou_scale_init", 0.3)),
            end=ou_scale_end,
            duration=explore_steps,
        )
        
        self.automatic_optimization = False
        self.sim_fn = mse_similarity

        # ---- Metrics/Rewards ----
        self.use_lpips = float(getattr(self.cfg.train, "lpips_lambda", 0.0)) > 0.0
        if self.use_lpips:
            try:
                import lpips
                net_name = getattr(self.cfg.train, "lpips_net", "vgg")
                self.lpips_net = lpips.LPIPS(net=net_name)
                for p in self.lpips_net.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"Warning: LPIPS failed to load: {e}")
                self.use_lpips = False
                self.lpips_net = None
        else:
            self.lpips_net = None

        self.use_msssim = float(getattr(self.cfg.train, "msssim_lambda", 0.0)) > 0.0

    # ---------------- helpers ----------------
    def _soft_update(self):
        tau = float(self.cfg.train.tau)
        target_nets.soft_update(self.actor_t, self.actor, tau)
        target_nets.soft_update(self.critic1_t, self.critic1, tau)
        target_nets.soft_update(self.critic2_t, self.critic2, tau)
        target_nets.soft_update(self.D_t, self.D, tau)

    def _init_canvas(self, I_star: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(I_star) 

    @staticmethod
    def _cond_pair(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, cond], dim=1)

    def _lpips_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if (not self.use_lpips) or (self.lpips_net is None):
            return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        d = self.lpips_net(x, y)
        return d.view(x.size(0))

    @staticmethod
    def _to01(x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5

    def _msssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.use_msssim:
            return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        x01 = self._to01(x).clamp(0, 1)
        y01 = self._to01(y).clamp(0, 1)
        score = ms_ssim(x01, y01, data_range=1.0, size_average=False, win_size=7)
        return score # [B]
    
    def _log_image(self, tag: str, img: torch.Tensor, step: int = None):
        logger = getattr(self, "logger", None)
        if logger is None: return

        if img.dim() == 3 and img.size(0) == 1:
            img_to_log = img.repeat(3, 1, 1)
        else:
            img_to_log = img
        step = int(step) if step is not None else int(self.global_step)

        if hasattr(logger, "log_image"):
            try:
                logger.log_image(key=tag, images=[img_to_log], step=step)
                return
            except Exception: pass

        exp = getattr(logger, "experiment", None)
        if exp is not None and hasattr(exp, "add_image"):
            try:
                exp.add_image(tag, img_to_log, global_step=step)
                return
            except Exception: pass

    # ---------------- optimizers ----------------
    def configure_optimizers(self):
        train = self.cfg.train
        # 建议在 config 中定义这个值，ViT 常规推荐 0.05
        wd = float(getattr(train, "weight_decay", 0.05)) 

        # 1. Actor, Renderer, Codebook 优化器
        pg = [
            {"params": self.actor.parameters(), "lr": float(train.actor_lr)},
            {"params": self.renderer.parameters(), "lr": float(getattr(train, "renderer_lr", train.actor_lr))},
        ]
        if hasattr(self.vq, "codebook") and len(list(self.vq.codebook.parameters())) > 0:
             pg.append({"params": self.vq.codebook.parameters(), "lr": float(getattr(train, "codebook_lr", train.actor_lr))})

        # 换成 AdamW
        opt_arc = torch.optim.AdamW(pg, weight_decay=wd)
        
        # 2. Critics 优化器 (ViT 架构的 Critic 同样受益于 AdamW)
        opt_critics = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(train.critic_lr),
            weight_decay=wd
        )
        
        # 3. Discriminator (判别器通常不需要太大的权重衰减，甚至可以保持 Adam)
        # 但为了风格统一，用 AdamW 并把 wd 设小一点或设为 0 也行
        opt_D = torch.optim.AdamW(
            self.D.parameters(), 
            lr=float(train.d_lr), 
            betas=(0.5, 0.999),
            weight_decay=0.0  # 判别器通常不建议设太强衰减
        )
        
        return [opt_arc, opt_critics, opt_D]

    def _compute_reward(self, I, C_curr, C_next, is_terminal):
        """
        统一的奖励计算逻辑。
        修正：修复了 AMP 训练下的类型匹配错误 (Float vs Half)。
        """
        T = self.cfg.train
        gamma = float(T.gamma)
        shaping_scale = float(T.shaping_scale)
        
        # 权重参数
        lambda_l2 = float(T.l2_reward_lambda)
        lambda_gan = float(T.adv_reward_lambda)
        lambda_lpips = float(T.lpips_lambda)
        lambda_msssim = float(T.msssim_lambda)

        # ==========================================
        # 1. 基础 MSE (L2) & Shaping (全程计算)
        # ==========================================
        mse_cur = F.mse_loss(C_curr, I, reduction="none").mean(dim=(1, 2, 3))
        mse_nxt = F.mse_loss(C_next, I, reduction="none").mean(dim=(1, 2, 3))
        
        # Shaping: R = (gamma * -MSE(s')) - (-MSE(s))
        r_shaping_l2 = ((gamma * (-mse_nxt)) - (-mse_cur)) * shaping_scale

        # ==========================================
        # 2. LPIPS 计算 & Shaping (全程计算)
        # ==========================================
        r_shaping_lpips = torch.zeros_like(r_shaping_l2)
        lp_nxt_cached = None 

        if self.use_lpips:
            lp_cur = self._lpips_dist(C_curr, I).view(-1)
            lp_nxt = self._lpips_dist(C_next, I).view(-1)
            lp_nxt_cached = lp_nxt 

            # LPIPS Shaping
            # 这里可能会涉及 float16 计算，但通常乘法运算会自动 broadcasting，
            # 为了保险起见，赋值给 r_shaping_lpips 时 PyTorch 通常能处理，
            # 但下面的索引赋值必须严格匹配。
            r_shaping_lpips = ((gamma * (-lp_nxt)) - (-lp_cur)) * shaping_scale

        # ==========================================
        # 3. 基础 Total Reward (由 Shaping 构成)
        # ==========================================
        total_reward = (lambda_l2 * r_shaping_l2) + (lambda_lpips * r_shaping_lpips)

        # ==========================================
        # 4. Terminal Bonus (叠加在最后一步)
        # ==========================================
        mask_idx = is_terminal.view(-1) # Boolean index

        if mask_idx.any():
            # 初始化 Terminal Bonus 容器
            # 这里的 zeros_like 会继承 r_shaping_l2 的类型 (通常是 Float32)
            bonus_l2 = torch.zeros_like(r_shaping_l2)
            bonus_gan = torch.zeros_like(r_shaping_l2)
            bonus_lpips = torch.zeros_like(r_shaping_l2)
            bonus_msssim = torch.zeros_like(r_shaping_l2)

            # --- A. L2 Absolute Score ---
            bonus_l2[mask_idx] = (-mse_nxt[mask_idx]) * shaping_scale

            # --- B. GAN Score ---
            # Clip WGAN scores to prevent unbounded Q-value targets.
            _set_requires_grad(self.D_t, False)
            gan_score = self.D_t(self._cond_pair(C_next[mask_idx], I[mask_idx]))
            _set_requires_grad(self.D_t, True)
            bonus_gan[mask_idx] = gan_score.view(-1).clamp(-10.0, 10.0).to(dtype=bonus_gan.dtype)

            # --- C. LPIPS Absolute Score ---
            if self.use_lpips:
                if lp_nxt_cached is not None:
                    # 【修复点】添加 .to(dtype=...) 
                    # 因为 lp_nxt_cached 可能是 float16，而 bonus_lpips 是 float32
                    val_to_assign = -lp_nxt_cached[mask_idx]
                    bonus_lpips[mask_idx] = val_to_assign.to(dtype=bonus_lpips.dtype)
                else:
                    lp_dist = self._lpips_dist(C_next[mask_idx], I[mask_idx])
                    bonus_lpips[mask_idx] = -lp_dist.view(-1).to(dtype=bonus_lpips.dtype)

            # --- D. MSSSIM Score ---
            if self.use_msssim:
                ms_score = self._msssim(C_next[mask_idx], I[mask_idx])
                bonus_msssim[mask_idx] = ms_score.to(dtype=bonus_msssim.dtype)

            # 叠加到 Total Reward
            total_reward[mask_idx] += (
                lambda_l2 * bonus_l2[mask_idx] + 
                lambda_gan * bonus_gan[mask_idx] + 
                lambda_lpips * bonus_lpips[mask_idx] + 
                lambda_msssim * bonus_msssim[mask_idx]
            )

        # ==========================================
        # 5. Log 信息
        # ==========================================
        info = {
            "mse_nxt": mse_nxt.detach(),
            "sim_score": (1.0 - mse_nxt.detach()),
            "r_shaping_l2": r_shaping_l2.detach(), 
            "r_shaping_lpips": r_shaping_lpips.detach(),
            "r_term_gan": torch.zeros_like(r_shaping_l2), 
            "mask_idx": mask_idx
        }
        if mask_idx.any():
            info["r_term_gan"][mask_idx] = bonus_gan[mask_idx].detach()

        return total_reward, info
    
    # ---------------- rollout ----------------
    def _run_rollout(self, batch, batch_idx: int) -> None:
        """收集一 batch 的 rollout 数据并写入 buffer，必要时打 log。"""
        T = self.cfg.train
        M = self.cfg.model
        I_star = batch["img"].to(self.device)
        B = I_star.size(0)

        C_live = self._init_canvas(I_star)
        horizon = int(T.horizon)
        stop_tau = float(T.stop_tau)
        warmup = int(T.warmup_steps)
        dump_every = int(getattr(T, "dump_every_n_batches", 200))
        do_dump = (batch_idx % dump_every == 0)
        debug_frames = []

        for t in range(horizon):
            t_norm = float(t) / max(float(horizon - 1), 1.0)
            t_emb = torch.full((B, 1), t_norm, device=self.device)

            if self.global_step < warmup:
                a = torch.randn(B, M.token_dim, device=self.device).tanh()
            else:
                a_det = self.actor.act_deterministic(I_star, C_live, t_emb=t_emb)
                ou_scale = self.ou_scale_schedule(self.global_step)
                a = (a_det + ou_scale * self.ou.sample_like(a_det)).clamp(-1.0, 1.0)

            _, z, _, _, _ = self.vq(a)
            C_next = self.renderer(C_live, z, t_emb=t_emb).clamp(-1, 1)

            sim_vec = self.sim_fn(C_next, I_star).view(B, -1).mean(dim=1)
            done_bool = (sim_vec >= stop_tau)
            done_float = done_bool.float().unsqueeze(1)

            self.buf.add(I_star, C_live, a, C_next, done=done_float, t=t_emb)
            C_live = C_next.detach()

            if do_dump and (t % 10 == 0):
                debug_frames.append(C_live[0].detach().cpu())

            if done_bool.all():
                break

        # Reset OU state at end of each episode to prevent cross-episode noise drift.
        self.ou.reset()

        if do_dump and len(debug_frames) > 0:
            debug_frames.append(I_star[0].detach().cpu())
            vis_tensor = torch.stack(debug_frames)
            vis_tensor = torch.nan_to_num(vis_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            grid = vutils.make_grid(
                vis_tensor,
                nrow=min(16, len(debug_frames)),
                normalize=True,
                value_range=(-1, 1),
            )
            self._log_image("rollout/anim", grid, self.global_step)

    def _step_discriminator(self) -> None:
        """用 buffer 中 terminal 样本训练判别器若干步。"""
        T = self.cfg.train
        M = self.cfg.model
        opt_arc, opt_critics, opt_D = self.optimizers()
        d_steps = int(T.d_steps)
        d_batch = int(T.d_batch)

        for _ in range(d_steps):
            if not self.buf.ready_terminal(d_batch):
                break
            real_I, fake_X = self.buf.sample_terminal(d_batch, self.device)

            self.toggle_optimizer(opt_D)
            opt_D.zero_grad(set_to_none=True)
            real_pair = self._cond_pair(real_I, real_I)
            fake_pair = self._cond_pair(fake_X, real_I)
            loss_D, _ = wgan_gp_loss(
                self.D, real=real_pair, fake=fake_pair,
                use_gp=bool(M.use_gp), gp_lambda=float(M.gp_lambda),
            )
            self.manual_backward(loss_D)
            self.clip_gradients(opt_D, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            opt_D.step()
            self.untoggle_optimizer(opt_D)
            self.log("loss/D", loss_D.detach(), prog_bar=True, on_step=True)

    def _step_critic(self) -> None:
        """从 buffer 采样，算 reward 与 target Q，更新双 Q。"""
        T = self.cfg.train
        opt_arc, opt_critics, opt_D = self.optimizers()
        batch_rl = int(T.batch_rl)
        gamma = float(T.gamma)

        if not self.buf.ready(batch_rl):
            return
        I_s, C_s, a_s, C_ns, done_s, t_s = self.buf.sample(batch_rl, self.device)
        is_terminal = (done_s > 0.5) | (t_s > 0.95)

        self.toggle_optimizer(opt_critics)
        opt_critics.zero_grad(set_to_none=True)
        with torch.no_grad():
            r_total, r_info = self._compute_reward(I_s, C_s, C_ns, is_terminal)
            v1_t = self.critic1_t(I_s, C_ns, t_emb=t_s)
            v2_t = self.critic2_t(I_s, C_ns, t_emb=t_s)
            min_v = torch.minimum(v1_t, v2_t)
            y = r_total + (1.0 - done_s.squeeze(1)) * gamma * min_v

        v1 = self.critic1(I_s, C_s, t_emb=t_s)
        v2 = self.critic2(I_s, C_s, t_emb=t_s)
        v_loss = F.smooth_l1_loss(v1, y) + F.smooth_l1_loss(v2, y)
        self.manual_backward(v_loss)
        self.clip_gradients(opt_critics, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_critics.step()
        self.untoggle_optimizer(opt_critics)

        self.log("loss/v_loss", v_loss.detach(), prog_bar=True)
        self.log("rew/total_mean", r_total.mean(), prog_bar=False)
        self.log("train/sim", r_info["sim_score"].mean(), prog_bar=True)

    def _step_actor(self) -> None:
        """DDPG：从 buffer 采样，确定性 Actor 得 C_next_hat，最大化 reward + gamma*Q，更新 Actor/Renderer 并软更新 target。"""
        T = self.cfg.train
        M = self.cfg.model
        opt_arc, opt_critics, opt_D = self.optimizers()
        batch_rl = int(T.batch_rl)
        gamma = float(T.gamma)

        if not self.buf.ready(batch_rl):
            return
        I_s, C_s, a_s, _, done_s, t_s = self.buf.sample(batch_rl, self.device)
        is_terminal = (done_s > 0.5) | (t_s > 0.95)

        self.toggle_optimizer(opt_arc)
        opt_arc.zero_grad(set_to_none=True)
        freeze_renderer = self.global_step < int(getattr(T, "renderer_freeze_steps", 0))
        _set_requires_grad(self.renderer, not freeze_renderer)

        a_hat = self.actor(I_s, C_s, t_emb=t_s)
        _, z_hat, commit_loss, codebook_loss, _ = self.vq(a_hat)
        C_next_hat = self.renderer(C_s, z_hat, t_emb=t_s).clamp(-1, 1)
        r_b_total, r_info_b = self._compute_reward(I_s, C_s, C_next_hat, is_terminal)

        self.log("train/sim_actor", r_info_b["sim_score"].mean(), prog_bar=False)

        v1_next = self.critic1_t(I_s, C_next_hat, t_emb=t_s)
        v2_next = self.critic2_t(I_s, C_next_hat, t_emb=t_s)
        min_v_next = torch.minimum(v1_next, v2_next)
        q_val = (1.0 - done_s.squeeze(1)) * min_v_next
        actor_obj = r_b_total + gamma * q_val
        loss_actor = -actor_obj.mean() + commit_loss + codebook_loss

        self.manual_backward(loss_actor)
        self.clip_gradients(opt_arc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_arc.step()
        self.untoggle_optimizer(opt_arc)
        self._soft_update()

        self.log("loss/actor", loss_actor.detach(), prog_bar=True)
        mask_idx = r_info_b["mask_idx"]
        if mask_idx.any():
            self.log("rew/term_gan_val", r_info_b["r_term_gan"][mask_idx].mean(), prog_bar=False)
        if (~mask_idx).any():
            self.log("rew/mid_shaping_l2", r_info_b["r_shaping_l2"][~mask_idx].mean(), prog_bar=False)
            if getattr(T, "lpips_lambda", 0.0) > 0:
                self.log("rew/mid_shaping_lpips", r_info_b["r_shaping_lpips"][~mask_idx].mean(), prog_bar=False)

    def training_step(self, batch, batch_idx):
        self._run_rollout(batch, batch_idx)
        self._step_discriminator()
        self._step_critic()
        self._step_actor()