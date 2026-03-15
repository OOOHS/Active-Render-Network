import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 基础组件 (AdaLN, SwiGLU, ViTBlock) - 与 Actor 保持一致
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
    pos = torch.arange(length, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(temperature) / embed_dim))
    pos_embed_val = pos.unsqueeze(1) * div_term.unsqueeze(0)
    pos_embed = torch.stack([pos_embed_val.sin(), pos_embed_val.cos()], dim=2).flatten(1)
    return pos_embed.unsqueeze(0) # [1, L, D]

# =====================================================================
# 3. Transformer Critic
# =====================================================================

class Critic(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()

        # ==================== Config Parsing ====================
        self.img_size = int(cfg_model.img_size)       # 64
        self.patch_size = int(cfg_model.patch_size)   # 8
        self.embed_dim = int(cfg_model.hidden_dim)    # 768
        
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size # 64
        
        depth = int(getattr(cfg_model, "critic_depth", 8)) # 建议与 Actor 接近
        num_heads = int(getattr(cfg_model, "critic_attn_heads", 12)) 
        cond_dim = 256

        # ==================== Embeddings ====================
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, 
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # Query Token (Value Token)
        self.query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
        
        # 共享位置编码: 1 + 64 = 65
        pos_embed_static = get_1d_sincos_pos_embed(self.embed_dim, 1 + self.num_patches)
        self.register_buffer("pos_embed", pos_embed_static)

        # 类型编码 (Target/Canvas/Query)
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

        # ==================== Head ====================
        # 不同于 Actor 的 mu/std，Critic 只输出一个标量 V
        self.v_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 2, 1)
        )

    def _patchify(self, img):
        return self.patch_embed(img).flatten(2).transpose(1, 2)

    def forward(self, I_star, C, t_emb=None):
        """
        I_star: [B, 3, 64, 64]
        C     : [B, 3, 64, 64]
        t_emb : [B, 1]
        """
        B = I_star.shape[0]
        if t_emb is None:
            t_emb = torch.zeros(B, 1, device=I_star.device)
            
        t_cond = self.t_mlp(t_emb)

        # 1. 构造序列 (逻辑与 Actor 完全一致)
        q = self.query_token.expand(B, -1, -1) + self.type_embed_query
        q = q + self.pos_embed[:, 0:1, :]

        tk_target = self._patchify(I_star) + self.type_embed_target
        tk_target = tk_target + self.pos_embed[:, 1:, :]

        tk_canvas = self._patchify(C) + self.type_embed_canvas
        tk_canvas = tk_canvas + self.pos_embed[:, 1:, :]

        x = torch.cat([q, tk_target, tk_canvas], dim=1) # [B, 1+2N, D]

        # 2. Transformer Backbone
        for block in self.blocks:
            x = block(x, t_cond)
        
        x = self.final_norm(x, t_cond)

        # 3. 提取特征并输出价值
        # 我们使用 Query Token (index 0) 汇总的信息来估计当前状态的价值
        v_feat = x[:, 0, :]
        v_out = self.v_head(v_feat) # [B, 1]

        return v_out.squeeze(1) # [B]