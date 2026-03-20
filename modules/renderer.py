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
# 3. Transformer Renderer
# =====================================================================

class Renderer(nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        
        # ==================== Config Parsing ====================
        self.canvas_ch = int(getattr(cfg_model, "canvas_channels", 3))
        self.img_size = int(cfg_model.img_size)      # 64
        self.patch_size = int(cfg_model.patch_size)  # 8
        self.token_dim = int(cfg_model.token_dim)    # 64
        self.embed_dim = int(cfg_model.hidden_dim)   # 768
        
        assert self.img_size % self.patch_size == 0
        self.grid_size = self.img_size // self.patch_size   # 8
        self.num_patches = self.grid_size * self.grid_size  # 64
        
        num_layers = int(getattr(cfg_model, "renderer_depth", 6))
        num_heads = int(getattr(cfg_model, "renderer_attn_heads", 12))
        cond_dim = 256
        
        # ==================== Layers ====================
        # 1. Time
        self.t_mlp = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim),
        )

        # 2. Input Projections
        # z (Action) -> embed_dim
        self.z_proj = nn.Linear(self.token_dim, self.embed_dim)
        
        # Canvas Patches -> embed_dim
        # 直接在这里定义，不再需要 _init_patch_helpers
        self.patch_embedder = nn.Conv2d(
            self.canvas_ch, self.embed_dim, 
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        # 3. Positional Embedding (Static)
        pos_embed_static = get_1d_sincos_pos_embed(self.embed_dim, 1 + self.num_patches)
        self.register_buffer("pos_embed", pos_embed_static)

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            ViTBlock(self.embed_dim, num_heads, cond_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = AdaLN(self.embed_dim, cond_dim)
        
        # 5. Output Head
        # 将 embed_dim 投影回 pixel 空间
        # 输出维度 = 3 * 8 * 8 = 192
        self.patch_dim = self.canvas_ch * (self.patch_size ** 2)
        self.head = nn.Linear(self.embed_dim, self.patch_dim)
        
        # Init Zero for output head (Good practice for regression)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def unpatchify(self, x):
        """
        [B, 64, 192] -> [B, 3, 64, 64]
        """
        B, N, _ = x.shape
        h = w = int(N**0.5) # 8
        p = self.patch_size # 8
        c = self.canvas_ch
        x = x.reshape(B, h, w, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        return x.reshape(B, c, h * p, h * p)

    def forward(self, C, z_embed, t_emb):
        """
        C: [B, 3, 64, 64]
        z_embed: [B, 64]
        t_emb: [B, 1]
        """
        t_cond = self.t_mlp(t_emb)
        
        # 1. Action Token [B, 1, 768]
        z_token = self.z_proj(z_embed).unsqueeze(1)
        
        # 2. Canvas Tokens [B, 64, 768]
        c_tokens = self.patch_embedder(C).flatten(2).transpose(1, 2)
        
        # 3. Concat & Add Pos
        x = torch.cat([z_token, c_tokens], dim=1)
        x = x + self.pos_embed

        # 4. Backbone
        for block in self.blocks:
            x = block(x, t_cond)

        x = self.final_norm(x, t_cond)
        
        # 5. Predict (Skip z_token)
        output_tokens = x[:, 1:, :] 
        pred_patches = self.head(output_tokens)
        
        # 6. Reconstruct as residual: C_next = C + delta
        # With zero-init head this means "no change" at init, which is the correct prior.
        delta = self.unpatchify(pred_patches)
        return C + delta