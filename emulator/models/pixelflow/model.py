from typing import Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import Timesteps, TimestepEmbedding, LabelEmbedding
import warnings

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    warnings.warn("`flash-attn` is not installed. Training mode may not work properly.", UserWarning)
    flash_attn_varlen_func = None


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs_cis.unbind(-1)
    if cos.ndim == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    else:
        cos = cos[:, None]
        sin = sin[:, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x_rotated = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]

    x_dtype = x.dtype
    x_rotated = torch.stack([x_rotated[1], -x_rotated[0]], dim=-1).flatten(3).to(x_dtype)
    out = (x * cos + x_rotated * sin).to(x_dtype)

    return out


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, bias=True, temporal=False):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=bias)
        self.temporal = temporal

    def forward_unfold(self, x):
        out_unfold = x.matmul(self.proj.weight.view(self.proj.weight.size(0), -1).t())
        if self.proj.bias is not None:
            out_unfold += self.proj.bias.to(out_unfold.dtype)
        return out_unfold

    # force fp32 for strict numerical reproducibility (debug only)
    # @torch.autocast('cuda', enabled=False)
    def forward(self, x):
        if self.training:
            return self.forward_unfold(x)
        if not self.temporal:
            out = self.proj(x)
            out = out.flatten(2).transpose(1, 2)  # BDHW -> BND
        else:
            # temporal patch embedding
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            out = self.proj(x) # (B*T, embed_dim, H/P, W/P)
            #  Get BND where N = T * (H/P) * (W/P)
            out = out.reshape(B, T, out.shape[1], out.shape[2], out.shape[3])  # (B, T, embed_dim, H/P, W/P)
            out = out.permute(0, 2, 1, 3, 4)  # (B, embed_dim, T, H/P, W/P)
            out = out.flatten(2)
            out = out.transpose(1, 2)

        return out

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, seqlen_list=None):
        input_dtype = x.dtype
        emb = self.linear(self.silu(timestep))

        if seqlen_list is not None:
            if x.shape[0] == timestep.shape[0]:
                # equivalent to `torch.repeat_interleave` but faster
                emb = torch.cat([one_emb[None].expand(repeat_time, -1) for one_emb, repeat_time in zip(emb, seqlen_list)])
            else:
                # timestep is (B , D).  Build emb sample-wise.
                emb_chunks = []
                for cond, n in zip(timestep, seqlen_list):   # B iterations
                    cond = cond.unsqueeze(0)                 # (1 , D)
                    emb_i = self.linear(self.silu(cond))     # (1 , 6D)
                    emb_chunks.append(emb_i.expand(n, -1))   # view, no alloc
                emb = torch.cat(emb_chunks, dim=0)           # still (L ,6D)
        else:
            emb = emb.unsqueeze(1)

        # Commented for memory
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.float().chunk(6, dim=-1)
        # x = self.norm(x).float() * (1 + scale_msa) + shift_msa
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa

        return x.to(input_dtype), gate_msa, shift_mlp, scale_mlp, gate_mlp
    

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, inner_dim=None, bias=True):
        super().__init__()
        inner_dim = int(dim * mult) if inner_dim is None else inner_dim
        dim_out = dim_out if dim_out is not None else dim
        self.fc1 = nn.Linear(dim, inner_dim, bias=bias)
        self.fc2 = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states =  F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(x_dtype)


class Attention(nn.Module):
    def __init__(self, q_dim, kv_dim=None, heads=8, head_dim=64, dropout=0.0, bias=False):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim if kv_dim is not None else q_dim
        self.inner_dim = head_dim * heads
        self.dropout = dropout
        self.head_dim = head_dim
        self.num_heads = heads

        self.q_proj = nn.Linear(self.q_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(self.kv_dim, self.inner_dim, bias=bias)

        self.o_proj = nn.Linear(self.inner_dim, self.q_dim, bias=bias)

        self.q_norm = RMSNorm(self.inner_dim)
        self.k_norm = RMSNorm(self.inner_dim)

    def prepare_attention_mask(
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py#L694
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ):
        head_size = self.num_heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def forward(
        self,
        inputs_q,
        inputs_kv,
        attention_mask=None,
        cross_attention=False,
        rope_pos_embed_q=None,
        rope_pos_embed_k=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ):

        inputs_kv = inputs_q if inputs_kv is None else inputs_kv
        query_states = self.q_proj(inputs_q)
        key_states = self.k_proj(inputs_kv)
        value_states = self.v_proj(inputs_kv)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if max_seqlen_q is None:
            assert not self.training, "PixelFlow needs sequence packing for training"

            bsz, q_len, _ = inputs_q.shape
            _, kv_len, _ = inputs_kv.shape

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

            query_states = apply_rotary_emb(query_states, rope_pos_embed_q)
            if not cross_attention:
                key_states = apply_rotary_emb(key_states, rope_pos_embed_q)
            else:
                if rope_pos_embed_k is not None:
                    key_states = apply_rotary_emb(key_states, rope_pos_embed_k)

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, kv_len, bsz)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(bsz, self.num_heads, -1, attention_mask.shape[-1])

            # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):  # strict numerical reproducibility (debug only)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.inner_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output

        else:
            # sequence packing mode
            query_states = query_states.view(-1, self.num_heads, self.head_dim)
            key_states = key_states.view(-1, self.num_heads, self.head_dim)
            value_states = value_states.view(-1, self.num_heads, self.head_dim)

            query_states = apply_rotary_emb(query_states.permute(1, 0, 2)[None], rope_pos_embed_q)[0].permute(1, 0, 2)
            if not cross_attention:
                key_states = apply_rotary_emb(key_states.permute(1, 0, 2)[None], rope_pos_embed_q)[0].permute(1, 0, 2)
            else:
                if rope_pos_embed_k is not None:
                    key_states = apply_rotary_emb(key_states.permute(1, 0, 2)[None], rope_pos_embed_k)[0].permute(1, 0, 2)

            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )

            attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, dropout=0.0,
        cross_attention_dim=None, attention_bias=False
    ):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim)

        # Self Attention
        self.attn1 = Attention(q_dim=dim, kv_dim=None, heads=num_attention_heads, head_dim=attention_head_dim, dropout=dropout, bias=attention_bias)

        if cross_attention_dim is not None:
            # Cross Attention
            self.norm2 = RMSNorm(dim, eps=1e-6)
            self.attn2 = Attention(q_dim=dim, kv_dim=cross_attention_dim, heads=num_attention_heads, head_dim=attention_head_dim, dropout=dropout, bias=attention_bias)
        else:
            self.attn2 = None

        self.norm3 = RMSNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        rope_pos_embed_q=None,
        rope_pos_embed_k=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqlen_list_q=None,
        seqlen_list_k=None,
        block_max_seqlen_q=None,
        block_max_seqlen_k=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, timestep, seqlen_list_q)

        if block_max_seqlen_q is not None:
            max_seqlen_q = block_max_seqlen_q
        elif seqlen_list_q is not None:
            max_seqlen_q = max(seqlen_list_q)
        else:
            max_seqlen_q = None

        if block_max_seqlen_k is not None:
            max_seqlen_k = block_max_seqlen_k
        elif seqlen_list_k is not None:
            max_seqlen_k = max(seqlen_list_k)
        else:
            max_seqlen_k = None

        attn_output = self.attn1(
            inputs_q=norm_hidden_states,
            inputs_kv=None,
            attention_mask=None,
            cross_attention=False,
            rope_pos_embed_q=rope_pos_embed_q,
            rope_pos_embed_k=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
        )

        attn_output_dtype = attn_output.dtype
        # attn_output = attn_output.float() ## Commented for memory
        attn_output = (gate_msa * attn_output).to(attn_output_dtype)
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                inputs_q=norm_hidden_states,
                inputs_kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                cross_attention=True,
                rope_pos_embed_q=rope_pos_embed_q,
                rope_pos_embed_k=rope_pos_embed_k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm3(hidden_states)
        norm_hidden_states_dtype = norm_hidden_states.dtype
        norm_hidden_states = norm_hidden_states.float()
        norm_hidden_states = (norm_hidden_states * (1 + scale_mlp) + shift_mlp).to(norm_hidden_states_dtype)
        norm_hidden_states = self.mlp(norm_hidden_states)
        ff_output = norm_hidden_states
        ff_output_dtype = ff_output.dtype
        ff_output = ff_output.float()
        ff_output = (gate_mlp * ff_output).to(ff_output_dtype)
        hidden_states = ff_output + hidden_states

        return hidden_states
    

class PixelFlowModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_attention_heads, attention_head_dim,
        depth, patch_size, dropout=0.0, cross_attention_dim=None, attention_bias=True, num_classes=0,
        temporal=False, use_grad_checkpointing=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attention_heads = num_attention_heads
        self.cross_attention_dim = cross_attention_dim
        self.patch_size = patch_size
        self.attention_head_dim = attention_head_dim
        self.num_classes = num_classes
        self.temporal = temporal

        embed_dim = num_attention_heads * attention_head_dim
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim, temporal=temporal)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embed_dim)

        # [stage] embedding
        self.latent_size_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embed_dim)
        if self.temporal:
            self.latent_time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embed_dim)
        if self.num_classes > 0:
            # class conditional
            self.class_embedder = LabelEmbedding(num_classes, embed_dim, dropout_prob=0.1)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim, attention_bias) for _ in range(depth)
        ])
        self.use_grad_checkpointing = use_grad_checkpointing

        self.norm_out = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(embed_dim, 2 * embed_dim)
        self.proj_out_2 = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        self.initialize_from_scratch()

    def initialize_from_scratch(self):
        print("Starting Initialization...")
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        nn.init.normal_(self.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.linear_2.weight, std=0.02)

        nn.init.normal_(self.latent_size_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.latent_size_embedder.linear_2.weight, std=0.02)

        if self.temporal:
            nn.init.normal_(self.latent_time_embedder.linear_1.weight, std=0.02)
            nn.init.normal_(self.latent_time_embedder.linear_2.weight, std=0.02)

        if self.num_classes > 0:
            nn.init.normal_(self.class_embedder.embedding_table.weight, std=0.02)

        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)

        nn.init.constant_(self.proj_out_1.weight, 0)
        nn.init.constant_(self.proj_out_1.bias, 0)
        nn.init.constant_(self.proj_out_2.weight, 0)
        nn.init.constant_(self.proj_out_2.bias, 0)

    def get_conditioning(self, timestep, dtype, class_labels=None, latent_size=None, latent_time=None):
        # timestep, class_embed, latent_size_embed
        timesteps_proj = self.time_proj(timestep)
        conditioning = self.timestep_embedder(timesteps_proj.to(dtype=dtype)) 

        if self.num_classes > 0:
            class_embed = self.class_embedder(class_labels)
            conditioning += class_embed

        if latent_size is not None:
            latent_size_proj = self.time_proj(latent_size)
            latent_size_embed = self.latent_size_embedder(latent_size_proj.to(dtype=dtype))
            conditioning += latent_size_embed

        if latent_time is not None:
            latent_time_proj = self.time_proj(latent_time)
            latent_time_embed = self.latent_time_embedder(latent_time_proj.to(dtype=dtype))
            conditioning += latent_time_embed

        return conditioning

    def shift_scale(self, hidden_states, conditioning, seqlen_list=None):
        shift, scale = self.proj_out_1(F.silu(conditioning)).float().chunk(2, dim=1)
        if seqlen_list is None:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        else:
            shift = torch.cat([shift_i[None].expand(ri, -1) for shift_i, ri in zip(shift, seqlen_list)])
            scale = torch.cat([scale_i[None].expand(ri, -1) for scale_i, ri in zip(scale, seqlen_list)])

        dtype = hidden_states.dtype
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * (1 + scale)
        hidden_states = hidden_states + shift
        hidden_states = hidden_states.to(dtype=dtype)

        # hidden_states = (self.norm_out(hidden_states).float() * (1 + scale) + shift).to(hidden_states.dtype)
        return hidden_states

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        class_labels=None,
        timestep=None,
        latent_size=None,
        latent_time=None,
        encoder_attention_mask=None,
        pos_embed_q=None,
        pos_embed_k=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqlen_list_q=None,
        seqlen_list_k=None,
    ):
        if self.temporal and seqlen_list_q is None:
            T = hidden_states.shape[1]
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        orig_height, orig_width = hidden_states.shape[-2], hidden_states.shape[-1]
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.patch_embed(hidden_states)

        conditioning = self.get_conditioning(
            timestep=timestep,
            dtype=hidden_states.dtype,
            class_labels=class_labels,
            latent_size=latent_size,
            latent_time=latent_time
        )

        for block in self.transformer_blocks:
            if self.use_grad_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    conditioning,
                    pos_embed_q,
                    pos_embed_k,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    seqlen_list_q,
                    seqlen_list_k
                )
            else:
                # Normal forward pass
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=conditioning,
                    rope_pos_embed_q=pos_embed_q,
                    rope_pos_embed_k=pos_embed_k,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    seqlen_list_q=seqlen_list_q,
                    seqlen_list_k=seqlen_list_k,
                )

        hidden_states = self.shift_scale(hidden_states, conditioning, seqlen_list_q)
        hidden_states = self.proj_out_2(hidden_states)
        if self.training:
            hidden_states = hidden_states.reshape(hidden_states.shape[0], self.patch_size, self.patch_size, self.out_channels)
            hidden_states = hidden_states.permute(0, 3, 1, 2).flatten(1)
            return hidden_states

        if self.temporal:
            # hidden states is (B, T*H*W, Hp*Wp*C)
            # reshape to (B, C, T, H, W)
            height, width = orig_height // self.patch_size, orig_width // self.patch_size
            hidden_states = hidden_states.reshape(
                shape=(-1, T, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nthwpqc->ntchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, T, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        else:
            # hidden states is (B, H*W, Hp*Wp*C)
            # reshape to (B, C, H, W)
            height, width = orig_height // self.patch_size, orig_width // self.patch_size
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        return output
