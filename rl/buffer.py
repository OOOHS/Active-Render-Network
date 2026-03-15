import torch
from typing import Tuple, Optional, Union

class ReplayBuffer:
    """
    双重缓冲机制：
    1. Main Buffer: 存所有过程数据 (用于 SAC 训练)
    2. Terminal Buffer: 只存终局数据 (用于 GAN/LPIPS 训练)
    """
    def __init__(
        self,
        capacity: int,
        term_capacity: int = 10000,  # 新增：终局缓冲区的容量
        img_shape: Optional[Tuple[int, int, int]] = None,
        action_dim: Optional[int] = None,
        dtype_img=torch.float32,
        dtype_act=torch.float32,
    ):
        self.capacity = int(capacity)
        self.term_capacity = int(term_capacity)
        
        # --- Main Buffer Pointers ---
        self.size = 0
        self.ptr = 0
        
        # --- Terminal Buffer Pointers ---
        self.term_size = 0
        self.term_ptr = 0

        self.dtype_img = dtype_img
        self.dtype_act = dtype_act

        self.img_shape = img_shape
        self.action_dim = action_dim

        # Main Storage
        self.I_buf = None
        self.C_buf = None
        self.a_buf = None
        self.Cn_buf = None
        self.done_buf = None
        self.t_buf = None

        # Terminal Storage (VIP)
        self.term_I_buf = None
        self.term_C_buf = None # 这里的 C 其实是 C_next (最终画作)
        # GAN 其实只需要 I (Real) 和 C_next (Fake)，但为了扩展性我们存全一点
        self.term_Cn_buf = None 
        
    def _ensure_storage(self, I, a):
        if self.I_buf is None:
            B, Cc, H, W = I.shape
            D = a.shape[1]
            self.img_shape = (Cc, H, W)
            self.action_dim = D

            def alloc(size, shape, dtype):
                return torch.empty((size, *shape), dtype=dtype, device="cpu")

            # Alloc Main
            self.I_buf    = alloc(self.capacity, (Cc, H, W), self.dtype_img)
            self.C_buf    = alloc(self.capacity, (Cc, H, W), self.dtype_img)
            self.Cn_buf   = alloc(self.capacity, (Cc, H, W), self.dtype_img)
            self.a_buf    = alloc(self.capacity, (D,),       self.dtype_act)
            self.done_buf = alloc(self.capacity, (1,),       torch.float32)
            self.t_buf    = alloc(self.capacity, (1,),       torch.float32)

            # Alloc Terminal (专门存 Real I 和 Fake Result C_next)
            self.term_I_buf  = alloc(self.term_capacity, (Cc, H, W), self.dtype_img)
            self.term_Cn_buf = alloc(self.term_capacity, (Cc, H, W), self.dtype_img)

    @torch.no_grad()
    def add(
        self,
        I: torch.Tensor,
        C: torch.Tensor,
        a: torch.Tensor,
        C_next: torch.Tensor,
        done: Optional[Union[torch.Tensor, float, int]] = None,
        t: Optional[Union[torch.Tensor, float, int]] = None,
    ):
        B = I.size(0)

        # ---- Normalize done & t to [B, 1] CPU ----
        if done is None:
            done_t = torch.zeros(B, 1, dtype=torch.float32, device="cpu")
        else:
            if not torch.is_tensor(done): done = torch.tensor(done, dtype=torch.float32)
            done = done.detach().cpu()
            if done.dim() == 0: done = done.repeat(B)
            if done.dim() == 1: done = done.unsqueeze(1)
            done_t = done.float()

        if t is None:
            t_t = torch.zeros(B, 1, dtype=torch.float32, device="cpu")
        else:
            if not torch.is_tensor(t): t = torch.tensor(t, dtype=torch.float32)
            t = t.detach().cpu()
            if t.dim() == 0: t = t.repeat(B)
            if t.dim() == 1: t = t.unsqueeze(1)
            t_t = t.float()

        self._ensure_storage(I, a)

        # Move data to CPU once
        I_cpu = I.detach().cpu()
        C_cpu = C.detach().cpu()
        Cn_cpu = C_next.detach().cpu()
        a_cpu = a.detach().cpu()

        for i in range(B):
            # 1. 存入 Main Buffer (用于 SAC)
            idx = self.ptr
            self.I_buf[idx].copy_(I_cpu[i])
            self.C_buf[idx].copy_(C_cpu[i])
            self.Cn_buf[idx].copy_(Cn_cpu[i])
            self.a_buf[idx].copy_(a_cpu[i])
            self.done_buf[idx].copy_(done_t[i])
            self.t_buf[idx].copy_(t_t[i])

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            # 2. 检查是否为终局，若是则存入 Terminal Buffer (用于 GAN)
            # 判定条件：done > 0.5 (早停) 或者 t > 0.95 (时间耗尽)
            is_terminal = (done_t[i] > 0.5) or (t_t[i] > 0.95)
            
            if is_terminal:
                t_idx = self.term_ptr
                self.term_I_buf[t_idx].copy_(I_cpu[i])
                self.term_Cn_buf[t_idx].copy_(Cn_cpu[i]) # 存的是 C_next (最终画)
                
                self.term_ptr = (self.term_ptr + 1) % self.term_capacity
                self.term_size = min(self.term_size + 1, self.term_capacity)

    def __len__(self):
        return self.size

    def ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
    
    def ready_terminal(self, batch_size: int) -> bool:
        """检查终局样本是否足够"""
        return self.term_size >= batch_size

    @torch.no_grad()
    def sample(self, batch_size: int, device: str = "cuda"):
        """普通采样 (SAC)"""
        assert self.size > 0
        idx = torch.randint(0, self.size, (batch_size,), device="cpu")

        return (
            self.I_buf[idx].to(device, non_blocking=True),
            self.C_buf[idx].to(device, non_blocking=True),
            self.a_buf[idx].to(device, non_blocking=True),
            self.Cn_buf[idx].to(device, non_blocking=True),
            self.done_buf[idx].to(device, non_blocking=True),
            self.t_buf[idx].to(device, non_blocking=True),
        )

    @torch.no_grad()
    def sample_terminal(self, batch_size: int, device: str = "cuda"):
        """专用采样：只返回 (Real I, Fake Final_Canvas)"""
        assert self.term_size > 0, "Terminal buffer is empty!"
        
        # 为了保证训练 D 的稳定性，如果不够 batch_size，允许重复采样
        # 但既然 ready_terminal 会检查，这里通常安全
        idx = torch.randint(0, self.term_size, (batch_size,), device="cpu")
        
        real = self.term_I_buf[idx].to(device, non_blocking=True)
        fake = self.term_Cn_buf[idx].to(device, non_blocking=True)
        return real, fake