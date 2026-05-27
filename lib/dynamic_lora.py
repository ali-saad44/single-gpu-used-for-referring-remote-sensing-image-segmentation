"""
Text-Guided Dynamic LoRA for Enhanced_RRSIS_UOT.

Instead of static LoRA adapters (same ΔW for every input), Dynamic LoRA
generates text-conditioned LoRA matrices so the vision encoder becomes
text-aware from the very first layer.

Key idea:
    - A small MLP (HyperNetwork) takes the pooled text embedding
    - It generates LoRA A and B matrices conditioned on the text
    - Each image sees a DIFFERENT set of adapter weights depending
      on what the referring expression asks for

This is our novel contribution: text-guided vision adaptation.

Reference:
    Inspired by HyperNetworks (Ha et al., 2016) and
    LoRA (Hu et al., 2021) — combined for cross-modal adaptation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a text-conditioned Dynamic LoRA adapter.
    
    Instead of generating massive A and B matrices directly from text (which costs 413M params),
    this uses a Modulated LoRA approach: A and B are static trainable parameters, 
    but the intermediate bottleneck is element-wise multiplied by a text-generated scale vector.

    Output = Original(x) + scaling * ((x @ A^T) * text_scale) @ B^T

    Args:
        original_linear: The frozen nn.Linear layer to wrap.
        text_dim: Dimension of text conditioning features.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
    """

    def __init__(self, original_linear, text_dim=256, rank=16, alpha=32.0):
        super().__init__()
        self.original_linear = original_linear
        self.scaling = alpha / rank
        self.rank = rank

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # Static LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, original_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Text modulation network (generates a scale vector of size 'rank')
        self.text_modulator = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.GELU(),
            nn.Linear(text_dim // 2, rank)
        )
        # Initialize text modulator to output 1.0 so it starts like standard LoRA
        nn.init.zeros_(self.text_modulator[-1].weight)
        nn.init.ones_(self.text_modulator[-1].bias)

        # Cached text features for current forward pass
        self._cached_text_feat = None

    def set_text_conditioning(self, text_feat):
        """
        Cache text features for the current forward pass.

        Args:
            text_feat: (B, text_dim) pooled text embedding.
        """
        self._cached_text_feat = text_feat

    def clear_text_conditioning(self):
        """Clear cached text features after forward pass."""
        self._cached_text_feat = None

    def forward(self, x):
        """
        Forward pass with text-modulated LoRA.

        Args:
            x: (*, in_features) input tensor.

        Returns:
            (*, out_features) output tensor.
        """
        base_out = self.original_linear(x)

        if self._cached_text_feat is not None:
            # Generate scaling vector from text: (B, rank)
            scale = self.text_modulator(self._cached_text_feat)

            # --- Fix for Windowed Attention in ViT ---
            # During windowed attention, input 'x' gets reshaped so its batch dimension
            # becomes B * num_windows. We must expand 'scale' to match it.
            B_x = x.shape[0]
            B_s = scale.shape[0]
            
            if B_x != B_s:
                if B_x % B_s == 0:
                    num_windows = B_x // B_s
                    # ViT window partition groups windows by image, so repeat_interleave is correct:
                    # [img1, img2] -> [img1_w1, img1_w2, img2_w1, img2_w2]
                    scale = scale.repeat_interleave(num_windows, dim=0)
                else:
                    # Fallback (should not happen in standard ViT)
                    scale = scale.mean(dim=0, keepdim=True).expand(B_x, -1)
            # ----------------------------------------

            if x.dim() == 3:
                # x: (B_w, N, D)
                hidden = F.linear(x, self.lora_A) # (B_w, N, rank)
                hidden = hidden * scale.unsqueeze(1) # Broadcast scale over sequence length N
                delta = F.linear(hidden, self.lora_B) # (B_w, N, out_features)
            elif x.dim() == 2:
                # x: (B_w, D)
                hidden = F.linear(x, self.lora_A) # (B_w, rank)
                hidden = hidden * scale # (B_w, rank)
                delta = F.linear(hidden, self.lora_B) # (B_w, out_features)
            else:
                # Fallback for other dimensions (e.g., 4D)
                hidden = F.linear(x, self.lora_A)
                # reshape scale to broadcast
                scale_view = scale.view(scale.shape[0], *([1]*(x.dim()-2)), scale.shape[-1])
                hidden = hidden * scale_view
                delta = F.linear(hidden, self.lora_B)

            return base_out + self.scaling * delta
        else:
            # Fallback (no text) - acts as standard static LoRA
            hidden = F.linear(x, self.lora_A)
            delta = F.linear(hidden, self.lora_B)
            return base_out + self.scaling * delta


class DynamicLoRAManager:
    """
    Manages text conditioning across all DynamicLoRALinear layers in a model.

    Call set_text_conditioning() before forward pass to propagate
    text features to all dynamic LoRA layers, and clear_text_conditioning()
    after the pass.
    """

    def __init__(self, model):
        self.dynamic_lora_layers = []
        for module in model.modules():
            if isinstance(module, DynamicLoRALinear):
                self.dynamic_lora_layers.append(module)

    def set_text_conditioning(self, text_feat):
        """Propagate text features to all dynamic LoRA layers."""
        for layer in self.dynamic_lora_layers:
            layer.set_text_conditioning(text_feat)

    def clear_text_conditioning(self):
        """Clear text features from all dynamic LoRA layers."""
        for layer in self.dynamic_lora_layers:
            layer.clear_text_conditioning()


def inject_dynamic_lora_adapters(model, text_dim=256, rank=16, alpha=32.0):
    """
    Inject Dynamic LoRA adapters into SAM3's ViT backbone.

    Replaces attention QKV and projection layers with text-conditioned
    dynamic LoRA wrappers.

    Args:
        model: The SAM3 model (Sam3Image).
        text_dim: Dimension of text features for conditioning.
        rank: LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Tuple of (num_params_added, DynamicLoRAManager).
    """
    lora_params = 0

    backbone = model.backbone
    if hasattr(backbone, 'vision_backbone'):
        vit = backbone.vision_backbone
        if hasattr(vit, 'trunk'):
            trunk = vit.trunk
            if hasattr(trunk, 'blocks'):
                for i, block in enumerate(trunk.blocks):
                    if hasattr(block, 'attn'):
                        attn = block.attn
                        # Dynamic LoRA on qkv projection
                        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
                            original = attn.qkv
                            attn.qkv = DynamicLoRALinear(
                                original, text_dim=text_dim,
                                rank=rank, alpha=alpha
                            )
                            lora_params += sum(
                                p.numel() for p in attn.qkv.parameters()
                                if p.requires_grad
                            )
                        # Dynamic LoRA on output projection
                        if hasattr(attn, 'proj') and isinstance(attn.proj, nn.Linear):
                            original = attn.proj
                            attn.proj = DynamicLoRALinear(
                                original, text_dim=text_dim,
                                rank=rank, alpha=alpha
                            )
                            lora_params += sum(
                                p.numel() for p in attn.proj.parameters()
                                if p.requires_grad
                            )

    manager = DynamicLoRAManager(model)

    print(f"[DynamicLoRA] Injected text-conditioned adapters (text_dim={text_dim}, rank={rank}, alpha={alpha})")
    print(f"[DynamicLoRA] Added {lora_params:,} trainable dynamic LoRA parameters")

    return lora_params, manager
