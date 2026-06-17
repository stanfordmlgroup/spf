import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List
from diffusers.configuration_utils import ConfigMixin, register_to_config

from emulator.models.pyramidal_flow.mmdit_modules.modeling_pyramid_mmdit import PyramidDiffusionMMDiT, EmbedNDRoPE
from emulator.models.pyramidal_flow.mmdit_modules.modeling_embedding import PatchEmbed3D, Timesteps, TimestepEmbedding
from emulator.models.pyramidal_flow.mmdit_modules.modeling_normalization import AdaLayerNormContinuous
from emulator.models.pyramidal_flow.mmdit_modules.modeling_mmdit_block import JointTransformerBlock
from emulator.models.pyramidal_flow.trainer_misc import is_sequence_parallel_initialized, get_sequence_parallel_group, get_sequence_parallel_world_size


class SpatiotemporalPyramidFlow(PyramidDiffusionMMDiT, ConfigMixin):
    """
    MMDiT (Pyramid) backbone with PixelFlow-style conditioning (class_labels, latent_size, latent_time).
    - Positional embeddings: applied by PatchEmbed3D exactly like Pyramid.
    - No historical conditioning: uses the given clips list as-is (earliest->latest), like Pyramid training.
    - Returns per-stage outputs [(B, C, T, H, W), ...].
    """

    @register_to_config
    def __init__(
        self,
        height: int,
        width: int,
        cond_height: int,
        cond_width: int,
        in_channels: int,
        cond_in_channels: int,
        out_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        depth: int,
        patch_size: int,
        cond_patch_size: int,
        max_num_frames: int = 200,
        qk_norm: str = 'rms_norm',
        pos_embed_type: str = 'rope',
        temp_pos_embed_type: str = 'sincos',
        temp_cond_pos_embed_type: str = 'sincos',
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        attention_bias: bool = True,   # kept for parity, not used
        num_classes: int = 0,
        temporal: bool = True,         # intended for (B, C, T, H, W) clips
        use_grad_checkpointing: bool = False,
        add_temp_pos_embed: bool = False,  # set True to add temporal sin/cos in PatchEmbed3D
        interp_condition_pos: bool = False,
        gradient_checkpointing_ratio: float = 0.6,
        pos_embed_max_size: int | None = None,      # pass-through to PatchEmbed3D; set if using "sincos",
        use_flash_attn: bool = True,          # whether to use flash attention where available
        use_temporal_causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.temporal = temporal
        self.num_classes = num_classes
        self.use_grad_checkpointing = use_grad_checkpointing

        self.inner_dim = num_attention_heads * attention_head_dim
        self.use_flash_attn = use_flash_attn
        self.use_temporal_causal = use_temporal_causal

        # --- Positional embeddings (Pyramid path). This is where pos_embed is applied. ---
        self.pos_embed_type = pos_embed_type
        self.temp_pos_embed_type = temp_pos_embed_type
        self.add_temp_pos_embed = add_temp_pos_embed
        self.pos_embed = PatchEmbed3D(
            height=height,
            width=width,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size if pos_embed_max_size is not None else max(height // patch_size, width // patch_size),
            max_num_frames=max_num_frames,
            pos_embed_type=pos_embed_type,        # spatial sin/cos + cropping/interp as in Pyramid
            temp_pos_embed_type=temp_pos_embed_type,   # temporal sin/cos is optional below
            add_temp_pos_embed=add_temp_pos_embed,
            interp_condition_pos=interp_condition_pos,
        )

        if temp_cond_pos_embed_type != 'sincos':
            raise ValueError(f"SpatiotemporalPyramidFlow only supports temp_cond_pos_embed_type='sincos', got {temp_cond_pos_embed_type}")
        self.cond_pos_embed = PatchEmbed3D(
            height=cond_height,
            width=cond_width,
            patch_size=cond_patch_size,
            in_channels=cond_in_channels,
            embed_dim=self.inner_dim,                      # produce tokens at same width as latent
            pos_embed_max_size=pos_embed_max_size if pos_embed_max_size is not None else max(cond_height // cond_patch_size, cond_width // cond_patch_size),
            max_num_frames=max_num_frames,
            pos_embed_type=pos_embed_type,
            temp_pos_embed_type=temp_cond_pos_embed_type,
            add_temp_pos_embed=add_temp_pos_embed,
            interp_condition_pos=interp_condition_pos,
        )
        # --- PixelFlow-style conditioning embedded into MMDiT temb ---
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
        self.latent_size_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
        self.latent_time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        class LabelEmbedding(nn.Module):
            def __init__(self, num_classes, embed_dim, dropout_prob=0.0):
                super().__init__()
                self.embedding_table = nn.Embedding(num_classes, embed_dim)
                self.dropout = nn.Dropout(dropout_prob)
            def forward(self, labels):
                return self.dropout(self.embedding_table(labels))

        self.class_embedder = LabelEmbedding(num_classes, self.inner_dim, dropout_prob=dropout) if num_classes > 0 else None

        # Optional cross-attention (context projected to inner_dim)
        self.context_embedder = nn.Linear(cross_attention_dim, self.inner_dim)

        if temp_pos_embed_type == 'rope':
            self.temp_rope_embed = EmbedNDRoPE(self.inner_dim, 10000, axes_dim=[attention_head_dim])  # NEW
        else:
            self.temp_rope_embed = None  # NEW

        # Transformer blocks (identical to Pyramid)
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=self.inner_dim,  # Pyramid uses full inner_dim here
                    qk_norm=qk_norm,
                    context_pre_only=(i == depth - 1),
                    use_flash_attn=self.use_flash_attn,
                )
                for i in range(depth)
            ]
        )

        # Head (Pyramid)
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, (patch_size * patch_size) * out_channels, bias=True)

        self.gradient_checkpointing = use_grad_checkpointing
        self.gradient_checkpointing_ratio = gradient_checkpointing_ratio

        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pos_embed.proj.bias, 0)

        for m in [self.timestep_embedder, self.latent_size_embedder, self.latent_time_embedder]:
            if m is not None:
                nn.init.normal_(m.linear_1.weight, std=0.02)
                nn.init.normal_(m.linear_2.weight, std=0.02)
        if self.class_embedder is not None:
            nn.init.normal_(self.class_embedder.embedding_table.weight, std=0.02)
        if self.context_embedder is not None:
            nn.init.normal_(self.context_embedder.weight, std=0.02)

        for blk in self.transformer_blocks:
            nn.init.constant_(blk.norm1.linear.weight, 0)
            nn.init.constant_(blk.norm1.linear.bias, 0)
            nn.init.constant_(blk.norm1_context.linear.weight, 0)
            nn.init.constant_(blk.norm1_context.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    # PixelFlow-style conditioning merged into Pyramid temb
    def _build_conditioning(
        self,
        timestep_ratio: torch.Tensor,
        dtype: torch.dtype,
        class_labels: torch.Tensor | None,
        latent_size: torch.Tensor | None,
        latent_time: torch.Tensor | None,
    ) -> torch.Tensor:
        t_proj = self.time_proj(timestep_ratio)
        temb = self.timestep_embedder(t_proj.to(dtype=dtype))

        if self.class_embedder is not None and class_labels is not None:
            temb.add_(self.class_embedder(class_labels))
        if latent_size is not None:
            ls_proj = self.time_proj(latent_size)
            temb.add_(self.latent_size_embedder(ls_proj.to(dtype=dtype)))
        if latent_time is not None:
            lt_proj = self.time_proj(latent_time)
            temb.add_(self.latent_time_embedder(lt_proj.to(dtype=dtype)))

        return temb

    def forward(
        self,
        sample: List[List[torch.Tensor]] | List[torch.Tensor] | torch.Tensor,
        condition_sample: List[List[torch.Tensor]] | List[torch.Tensor] | torch.Tensor,
        timestep_ratio: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        latent_size: torch.Tensor | None = None,
        latent_time: torch.Tensor | None = None,
    ):
        # Infer num_stages and pad batch size from the latent `sample`
        num_stages = len(sample)
        if isinstance(sample[0], list):
            pad_batch_size = sample[0][-1].shape[0]
        else:
            pad_batch_size = sample[0].shape[0]

        # ------------------------------------------------------------------------------
        # Patchify + position the encoder stream with PatchEmbed3D (stage-wise)
        #      Accepts the same nested structure as `sample`.
        # ------------------------------------------------------------------------------
        enc_tok_list = self.cond_pos_embed(condition_sample)  # list[(B_s, S_enc_i, D)] or Tensor
        if isinstance(enc_tok_list, torch.Tensor):
            enc_tok_list = [enc_tok_list]  # normalize to list per stage

        assert len(enc_tok_list) == num_stages, \
            f"Encoder stages ({len(enc_tok_list)}) must match latent stages ({num_stages})."

        # Stage-wise encoder lengths (can be unequal)
        enc_lens = [x.shape[1] for x in enc_tok_list]
        enc_len_max = max(enc_lens)

        # Build the big encoder token table ctx: (B_total, enc_len_max, D), arranged so
        # rows i, i+num_stages, ... belong to stage i (matching Pyramid's i::num_stages pattern).
        B_total = pad_batch_size * num_stages
        device = enc_tok_list[0].device
        dtype = enc_tok_list[0].dtype

        with torch.no_grad():
            ctx = torch.zeros(B_total, enc_len_max, self.inner_dim, device=device, dtype=dtype)  # NEW
            enc_mask = torch.zeros(B_total, enc_len_max, device=device, dtype=torch.long)        # NEW

            for i_s, enc_tokens in enumerate(enc_tok_list):
                S_i = enc_tokens.shape[1]
                ctx[i_s::num_stages, :S_i, :].copy_(enc_tokens)
                enc_mask[i_s::num_stages, :S_i] = 1
        del enc_tok_list

        # ------------------------------------------------------------------------------
        # Merge latent inputs (applies PatchEmbed3D + masks exactly like Pyramid)
        # We pass the max encoder length + a per-row mask we just built.
        # ------------------------------------------------------------------------------
        hidden_states, hidden_length, temps, heights, widths, trainable_token_list, \
            enc_attn_mask_pkg, attn_mask_4d, image_rope = self.merge_input(   # unchanged helper
                sample=sample,
                encoder_hidden_length=enc_len_max,
                encoder_attention_mask=enc_mask,   # internally converted to varlen indices
            )

        if not isinstance(hidden_states, torch.Tensor):
            hidden_states = torch.cat(hidden_states, dim=1)  # (pad_batch_size, sum_stage_tokens, D)

        # Global conditioning (unchanged)
        temb = self._build_conditioning(
            timestep_ratio=timestep_ratio,
            dtype=hidden_states.dtype,
            class_labels=class_labels,
            latent_size=latent_size,
            latent_time=latent_time,
        )

        # ------------------------------------------------------------------------------
        # Transformer blocks (identical scheduling as Pyramid)
        # Note: `ctx` has shape (B_total, enc_len_max, D); hidden_states has (pad_B, ...).
        # Pyramid's varlen attention uses i::num_stages row pattern to align them.
        # ------------------------------------------------------------------------------
        del sample, condition_sample
        for i_b, block in enumerate(self.transformer_blocks):
            if self.training and self.use_grad_checkpointing and (i_b >= int(len(self.transformer_blocks) * self.gradient_checkpointing_ratio)):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                ctx, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    ctx,                      # encoder tokens (B_total, enc_len_max, D)
                    enc_attn_mask_pkg,        # varlen indices built internally
                    temb,
                    attn_mask_4d,
                    hidden_length,
                    image_rope,
                    use_reentrant=False,
                )
            else:
                ctx, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=ctx,         # CHANGED: our stage-aware encoder tokens
                    encoder_attention_mask=enc_attn_mask_pkg,  # CHANGED: built internally from enc_mask
                    temb=temb,
                    attention_mask=attn_mask_4d,
                    hidden_length=hidden_length,
                    image_rotary_emb=image_rope,
                )

        hidden_states = self.norm_out(hidden_states, temb, hidden_length=hidden_length)
        hidden_states = self.proj_out(hidden_states)

        # Split back into per-stage outputs (B, C, T, H, W)
        output = self.split_output(hidden_states, hidden_length, temps, heights, widths, trainable_token_list)
        return output
