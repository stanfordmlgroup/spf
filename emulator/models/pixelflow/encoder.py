import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional
from einops import rearrange


class PixelFlowEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 5,
        patch_size: int | tuple[int, int] = 16,
        model_dim: int = 1152,
        flatten_first: bool = True,
    ):
        super().__init__()

        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.patch_size = patch_size
        self.model_dim = model_dim

        # 1. Patchify with a conv (kernel = stride = patch_size)
        self.proj = nn.Conv2d(
            in_channels=num_channels,
            out_channels=model_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. Optionally flatten patches to (B, N, embed_dim)
        self.flatten_first = flatten_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns: (B, N, embed_dim)
        """
        B, C, H, W = x.shape
        h_p, w_p   = H // self.patch_size[0], W // self.patch_size[1]

        x = self.proj(x)                        # (B, embed_dim, H/ph, W/pw)

        if self.flatten_first:
            x = x.flatten(2)                    # (B, embed_dim, N)
            x = x.transpose(1, 2).contiguous()  # (B, N, embed_dim)

        return x


class TemporalPixelFlowEncoder(nn.Module):
    """
    Patch-embedder for conditioning inputs used with *TemporalPixelFlow*.

    -------------------------------------------------------------------------
    • **Training  (sequence-packed)**
        forward(tokens, seqlen_q, seqlen_q_t)
            tokens     : (L , D_in)     – already patchified & (optionally)
                                          projected; L = Σ seqlen_q
            seqlen_q   : list[int]      – (#spatial·temporal patches / sample)
            seqlen_q_t : list[int]      – (#frames / sample)

        returns tokens_out, seqlen_q, seqlen_q_t
            tokens_out : (L , model_dim)

    • **Eval / inference  (dense)**
        forward(images)
            images     : (B,C,H,W) or (B,T,C,H,W)

        returns encodings : (B, T·N, model_dim)
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        num_channels: int = 5,
        patch_size: int | Tuple[int, int] = 16,
        model_dim: int = 1152,
    ):
        super().__init__()

        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.p_h, self.p_w = patch_size
        self.model_dim = model_dim
        self.in_dim = num_channels * self.p_h * self.p_w

        # shared projection (C → D)
        self.linear = nn.Linear(self.in_dim, model_dim)

    def _forward_packed(
        self,
        tokens: torch.Tensor,           # (L , Din)
    ):
        tokens = self.linear(tokens)

        return tokens

    def _forward_dense(self, imgs: torch.Tensor):
        """
        imgs : (B,T,C,H,W)
        out  : (B, T·N, D)
        """
        B, T, C, H, W = imgs.shape
        assert H % self.p_h == 0 and W % self.p_w == 0, "H, W must be multiples of patch size"

        # (B,T,C,H,W) -> (B, T·H/p·W/p, C·p²)
        x = rearrange(imgs, 'b t c (h ph) (w pw) -> b (t h w) (c ph pw)', ph=self.p_h, pw=self.p_w)

        # (B, T·H/p·W/p, C·p²) -> (B, T·H/p·W/p, D)
        x = self.linear(x)                                

        return x

    # ------------------------------------------------------------------ #
    # unified forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, List[int], List[int]]],
    ):
        """
        • If **training**: supply  packed tokens
        • If **eval**   : supply  raw images
        """
        if self.training:
            return self._forward_packed(x)

        # evaluation / inference
        return self._forward_dense(x)
