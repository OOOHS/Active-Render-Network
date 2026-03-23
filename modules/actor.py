import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 基础组件 (AdaLN, SwiGLU, ViTBlock)
# =====================================================================

class AdaLN(nn.Module):
    def __init__(self, num_channels: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, elementwise_affine=False, eps=eps)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, num_channels * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, t_emb):
        res = self.linear(self.silu(t_emb))
        gamma, beta = res.chunk(2, dim=1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return self.norm(x) * (1 + gamma) + beta

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = AdaLN(dim, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = AdaLN(dim, cond_dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, t_emb):
        h = self.norm1(x, t_emb)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x, t_emb))
        return x

# =====================================================================
# 2. 辅助函数: 正余弦位置编码
# =====================================================================

def get_1d_sincos_pos_embed(embed_dim, length, temperature=10000.0):
    """
    生成固定的 1D 正余弦位置编码
    """
    pos = torch.arange(length, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(temperature) / embed_dim))
    pos_embed_val = pos.unsqueeze(1) * div_term.unsqueeze(0)
    
    pos_embed = torch.stack([pos_embed_val.sin(), pos_embed_val.cos()], dim=2).flatten(1)
    return pos_embed.unsqueeze(0) # [1, L, D]

# =====================================================================
# 3. ViT Actor
# =====================================================================

class Actor(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()

        # ==================== Config Parsing ====================
        # 严格按照 yaml 读取尺寸
        self.img_size = int(cfg_model.img_size)       # 64
        self.patch_size = int(cfg_model.patch_size)   # 8
        self.token_dim = int(cfg_model.token_dim)     # 64
        self.embed_dim = int(cfg_model.hidden_dim)    # 768
        
        # 计算网格信息
        assert self.img_size % self.patch_size == 0, \
            f"Img size {self.img_size} must be divisible by patch size {self.patch_size}"
        
        self.grid_size = self.img_size // self.patch_size  # 64/8 = 8
        self.num_patches = self.grid_size * self.grid_size # 8*8 = 64
        
        # 校验 seq_len (可选，确保 config 一致性)
        if hasattr(cfg_model, "seq_len"):
            assert self.num_patches == int(cfg_model.seq_len), \
                f"Calculated patches ({self.num_patches}) != config seq_len ({cfg_model.seq_len})"

        depth = int(getattr(cfg_model, "actor_depth", 8))
        num_heads = int(getattr(cfg_model, "actor_attn_heads", 12)) 
        cond_dim = 256 # 时间嵌入维度，通常固定

        # ==================== Embeddings ====================
        # Patch Embedder: [3, 8, 8] -> [768]
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, 
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # Query Token (Intent)
        self.query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        
        # 固定正余弦位置编码: 长度 = 1 (Query) + 64 (Patches) = 65
        pos_embed_static = get_1d_sincos_pos_embed(self.embed_dim, 1 + self.num_patches)
        self.register_buffer("pos_embed", pos_embed_static)

        # Type Embeddings
        self.type_embed_target = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.type_embed_canvas = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.type_embed_query  = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        nn.init.trunc_normal_(self.type_embed_target, std=0.02)
        nn.init.trunc_normal_(self.type_embed_canvas, std=0.02)
        nn.init.trunc_normal_(self.type_embed_query, std=0.02)

        # ==================== Backbone ====================
        self.t_mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList([
            ViTBlock(self.embed_dim, num_heads, cond_dim)
            for _ in range(depth)
        ])
        self.final_norm = AdaLN(self.embed_dim, cond_dim)

        # ==================== Heads (SAC：μ + log σ) ====================
        self.mu_head = nn.Linear(self.embed_dim, self.token_dim)

        self.log_std_min = float(getattr(cfg_model, "actor_log_std_min", -5.0))
        self.log_std_max = float(getattr(cfg_model, "actor_log_std_max", 2.0))
        self.use_std_head = bool(getattr(cfg_model, "actor_std_head", True))

        if self.use_std_head:
            self.logstd_head = nn.Linear(self.embed_dim, self.token_dim)
        else:
            self.log_std = nn.Parameter(torch.full((self.token_dim,), -0.5))

    def _patchify(self, img):
        # [B, 3, 64, 64] -> [B, 768, 8, 8] -> [B, 64, 768]
        return self.patch_embed(img).flatten(2).transpose(1, 2)

    def _forward_backbone(self, I_star, C, t_emb):
        B = I_star.shape[0]
        t_cond = self.t_mlp(t_emb) 

        # A. Query (Type Q + Pos 0)
        q = self.query_token.expand(B, -1, -1) + self.type_embed_query
        q = q + self.pos_embed[:, 0:1, :]

        # B. Target (Type T + Pos 1..N)
        tk_target = self._patchify(I_star) + self.type_embed_target
        tk_target = tk_target + self.pos_embed[:, 1:, :] 

        # C. Canvas (Type C + Pos 1..N) - 共享位置编码
        tk_canvas = self._patchify(C) + self.type_embed_canvas
        tk_canvas = tk_canvas + self.pos_embed[:, 1:, :] 

        # [Query, Target, Canvas] -> Length = 1 + 64 + 64 = 129
        x = torch.cat([q, tk_target, tk_canvas], dim=1)

        for block in self.blocks:
            x = block(x, t_cond)
        
        x = self.final_norm(x, t_cond)
        return x[:, 0, :] # 返回 Query Token

    def _get_mu_logstd(self, I_star, C, t_emb=None):
        if t_emb is None:
            t_emb = torch.zeros(I_star.shape[0], 1, device=I_star.device)
        h = self._forward_backbone(I_star, C, t_emb)
        mu = self.mu_head(h)
        if self.use_std_head:
            log_std = self.logstd_head(h)
        else:
            log_std = self.log_std.unsqueeze(0).expand_as(mu)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    @torch.no_grad()
    def act_deterministic(self, I_star, C, t_emb=None):
        mu, _ = self._get_mu_logstd(I_star, C, t_emb)
        return torch.tanh(mu)

    def forward(self, I_star, C, t_emb=None):
        mu, _ = self._get_mu_logstd(I_star, C, t_emb)
        return torch.tanh(mu)

    def sample(self, I_star, C, t_emb=None):
        eps = 1e-6
        mu, log_std = self._get_mu_logstd(I_star, C, t_emb)
        std = torch.exp(log_std)
        noise = torch.randn_like(mu)
        u = mu + std * noise
        a = torch.tanh(u)

        logp_gauss = -0.5 * (((u - mu) / (std + eps)) ** 2 + 2 * log_std + math.log(2 * math.pi))
        logp_gauss = logp_gauss.sum(dim=1)
        logp_correction = torch.log(1 - a.pow(2) + eps).sum(dim=1)
        logp = logp_gauss - logp_correction

        return a, logp, mu, log_std