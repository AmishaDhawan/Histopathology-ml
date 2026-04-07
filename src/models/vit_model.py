"""
Vision Transformer (ViT-Base) with FlashAttention toggle.

Architecture decisions vs ResNet-50:
=====================================
ResNet-50 uses convolutional layers with LOCAL receptive fields.
Even with global average pooling at the end, each prediction is dominated
by local texture around that spatial location.

A CD8+ T cell looks identical to a ResNet whether it is:
- 50 microns from a tumor boundary (clinically significant: immune attack)
- 2mm away from any tumor (clinically insignificant: background immune cell)

ViT uses self-attention: every patch token attends to every other patch token.
This means the representation of any patch is EXPLICITLY conditioned on all
other patches in the image -- enabling global spatial reasoning.

Why ViT-Base and not Swin Transformer:
=======================================
Swin Transformer uses SHIFTED WINDOW attention. Within each window, tokens
attend to each other. Across windows, interaction is limited to the shifted
overlap. This restricts long-range spatial interactions for memory efficiency.

Our biological task REQUIRES global attention: we need to relate a tumor
cell cluster in the top-left to an immune infiltrate in the bottom-right.
Swin's window structure would prevent this. Standard ViT with global
self-attention is the correct choice, at the cost of higher memory usage.
(We solve the memory problem with FlashAttention, not by using Swin.)

Pretraining:
============
Initialize from ViT-Base/16 pretrained with DINO on ImageNet.
DINO (self-supervised) produces better features than supervised ImageNet
for downstream fine-tuning -- shown consistently in pathology benchmarks
(CONCH Nature Medicine 2024, UNI Nature Medicine 2024).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FlashAttention(nn.Module):
    """
    Multi-head self-attention with optional FlashAttention backend.

    When use_flash=True, delegates to F.scaled_dot_product_attention which
    automatically selects the FlashAttention v2 kernel when running on
    compatible hardware (Ampere+ GPUs with CUDA). On CPU or older GPUs,
    PyTorch falls back to a mathematically equivalent implementation.

    When use_flash=False, computes standard dot-product attention with an
    explicit softmax, materializing the full n x n attention matrix in HBM.
    This path is used when we need to extract attention weights for
    visualization (FlashAttention never materializes the attention matrix).

    Args:
        embed_dim: total embedding dimension
        num_heads: number of attention heads
        use_flash: if True, use F.scaled_dot_product_attention
    """

    def __init__(self, embed_dim: int, num_heads: int, use_flash: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash = use_flash

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Store attention weights for visualization (only when use_flash=False)
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, N, D) where N = num_patches + 1 (CLS)

        Returns:
            output tensor of shape (B, N, D)
        """
        B, N, D = x.shape

        # Compute Q, K, V projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        if self.use_flash:
            # F.scaled_dot_product_attention automatically uses FlashAttention v2
            # when available (Ampere+ GPU with CUDA). On CPU or older hardware,
            # PyTorch uses a mathematically equivalent fallback.
            # attn_mask=None and is_causal=False because ViT uses bidirectional
            # attention (every patch attends to every other patch).
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=False
            )
            self._attn_weights = None  # FlashAttention never materializes weights
        else:
            # Standard attention: explicitly compute and store the n x n matrix
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale  # (B, heads, N, N)
            attn = attn.softmax(dim=-1)
            self._attn_weights = attn.detach()  # Store for visualization
            out = attn @ v  # (B, heads, N, head_dim)

        # Reshape back to (B, N, D)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Return stored attention weights from the last forward pass.

        Only available when use_flash=False. Returns None when FlashAttention
        is enabled because it never materializes the attention matrix.

        Returns:
            Tensor of shape (B, num_heads, N, N) or None
        """
        return self._attn_weights


class TransformerBlock(nn.Module):
    """
    Standard ViT transformer block: LayerNorm -> MSA -> LayerNorm -> MLP.

    Uses pre-norm (LayerNorm before attention and MLP) following the
    original ViT paper and DINO. Pre-norm is more stable for training
    than post-norm, especially for deeper transformers.

    Args:
        embed_dim: embedding dimension
        num_heads: number of attention heads
        mlp_ratio: ratio of MLP hidden dim to embed_dim (default 4.0)
        dropout: dropout rate
        use_flash: whether to use FlashAttention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_flash: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FlashAttention(embed_dim, num_heads, use_flash=use_flash)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTHistoClassifier(nn.Module):
    """
    Vision Transformer for histopathology tile classification.

    Architecture decisions vs ResNet-50:
    =====================================
    ResNet-50 uses convolutional layers with LOCAL receptive fields.
    Even with global average pooling at the end, each prediction is dominated
    by local texture around that spatial location.

    A CD8+ T cell looks identical to a ResNet whether it is:
    - 50 microns from a tumor boundary (clinically significant: immune attack)
    - 2mm away from any tumor (clinically insignificant: background immune cell)

    ViT uses self-attention: every patch token attends to every other patch token.
    This means the representation of any patch is EXPLICITLY conditioned on all
    other patches in the image -- enabling global spatial reasoning.

    Why ViT-Base and not Swin Transformer:
    =======================================
    Swin Transformer uses SHIFTED WINDOW attention. Within each window, tokens
    attend to each other. Across windows, interaction is limited to the shifted
    overlap. This restricts long-range spatial interactions for memory efficiency.

    Our biological task REQUIRES global attention: we need to relate a tumor
    cell cluster in the top-left to an immune infiltrate in the bottom-right.
    Swin's window structure would prevent this. Standard ViT with global
    self-attention is the correct choice, at the cost of higher memory usage.
    (We solve the memory problem with FlashAttention, not by using Swin.)

    Pretraining:
    ============
    Initialize from ViT-Base/16 pretrained with DINO on ImageNet.
    DINO (self-supervised) produces better features than supervised ImageNet
    for downstream fine-tuning -- shown consistently in pathology benchmarks
    (CONCH Nature Medicine 2024, UNI Nature Medicine 2024).

    Args:
        num_classes: number of output classes
        use_flash_attention: if True, use FlashAttention v2 via
                             F.scaled_dot_product_attention (PyTorch 2.0+)
        patch_size: 16 for cellular resolution, 32 for tissue-level only
        image_size: 224 for efficient training, 1024 for full resolution
        pretrained: whether to load pretrained weights
        dropout: dropout rate for transformer blocks and classifier head
    """

    # ViT-Base/16 architecture constants
    EMBED_DIM = 768
    NUM_HEADS = 12
    NUM_LAYERS = 12
    MLP_RATIO = 4.0

    def __init__(
        self,
        num_classes: int = 4,
        use_flash_attention: bool = True,
        patch_size: int = 16,
        image_size: int = 224,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_flash_attention = use_flash_attention
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = self.EMBED_DIM
        self.num_heads = self.NUM_HEADS

        # Number of patches: (image_size / patch_size)^2
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding: Conv2d projects each patch_size x patch_size region
        # into a D-dimensional vector. This is equivalent to linear projection
        # of flattened patches but more efficient.
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.EMBED_DIM,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token: a learnable token prepended to the patch sequence.
        # Its final representation is used for classification.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.EMBED_DIM))

        # Positional embeddings: learnable, one per patch + CLS token.
        # Without these, the model has no notion of spatial arrangement.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.EMBED_DIM)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.EMBED_DIM,
                num_heads=self.NUM_HEADS,
                mlp_ratio=self.MLP_RATIO,
                dropout=dropout,
                use_flash=use_flash_attention,
            )
            for _ in range(self.NUM_LAYERS)
        ])

        self.norm = nn.LayerNorm(self.EMBED_DIM)

        # Classification head: LayerNorm -> Linear
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.EMBED_DIM, num_classes),
        )

        # Initialize weights
        self._init_weights(pretrained)

    def _init_weights(self, pretrained: bool):
        """
        Initialize model weights.

        If pretrained=True, load ViT-Base/16 ImageNet weights from torchvision
        and transfer compatible parameters. The classification head is always
        randomly initialized since our task has different num_classes.
        """
        # Initialize positional embeddings and CLS token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if pretrained:
            self._load_pretrained_weights()
        else:
            # Xavier uniform for all linear layers
            self.apply(self._init_weights_fn)

    @staticmethod
    def _init_weights_fn(m: nn.Module):
        """Xavier uniform initialization for linear and conv layers."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _load_pretrained_weights(self):
        """
        Load pretrained ViT-Base/16 weights from torchvision.

        Uses torchvision's ViT_B_16_Weights.IMAGENET1K_V1. We transfer:
        - Patch embedding (conv_proj)
        - Positional embeddings
        - CLS token
        - All transformer encoder blocks
        The classification head is NOT transferred (different num_classes).
        """
        pretrained_vit = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        # Transfer patch embedding weights
        self.patch_embed.weight.data.copy_(pretrained_vit.conv_proj.weight.data)
        self.patch_embed.bias.data.copy_(pretrained_vit.conv_proj.bias.data)

        # Transfer CLS token and positional embeddings
        # torchvision ViT stores class_token as (1, 1, D)
        self.cls_token.data.copy_(pretrained_vit.class_token.data)

        # Positional embeddings: torchvision stores as (1, N+1, D)
        # If our image_size differs from 224, we'd need to interpolate.
        # For image_size=224, patch_size=16: num_patches=196, same as pretrained.
        pretrained_pos = pretrained_vit.encoder.pos_embedding.data
        if pretrained_pos.shape == self.pos_embed.shape:
            self.pos_embed.data.copy_(pretrained_pos)
        else:
            # Interpolate positional embeddings for different image sizes
            self._interpolate_pos_embed(pretrained_pos)

        # Transfer transformer encoder blocks
        for i, block in enumerate(self.blocks):
            src_block = pretrained_vit.encoder.layers[i]

            # LayerNorm 1
            block.norm1.weight.data.copy_(src_block.ln_1.weight.data)
            block.norm1.bias.data.copy_(src_block.ln_1.bias.data)

            # Self-attention: torchvision uses in_proj_weight/bias for QKV
            block.attn.qkv.weight.data.copy_(
                src_block.self_attention.in_proj_weight.data
            )
            block.attn.qkv.bias.data.copy_(
                src_block.self_attention.in_proj_bias.data
            )
            block.attn.proj.weight.data.copy_(
                src_block.self_attention.out_proj.weight.data
            )
            block.attn.proj.bias.data.copy_(
                src_block.self_attention.out_proj.bias.data
            )

            # LayerNorm 2
            block.norm2.weight.data.copy_(src_block.ln_2.weight.data)
            block.norm2.bias.data.copy_(src_block.ln_2.bias.data)

            # MLP: torchvision >= 0.14 MLPBlock inherits from nn.Sequential
            # and uses indexed children: mlp[0] = first Linear, mlp[3] = second Linear
            block.mlp[0].weight.data.copy_(src_block.mlp[0].weight.data)
            block.mlp[0].bias.data.copy_(src_block.mlp[0].bias.data)
            block.mlp[3].weight.data.copy_(src_block.mlp[3].weight.data)
            block.mlp[3].bias.data.copy_(src_block.mlp[3].bias.data)

        # Transfer final LayerNorm
        self.norm.weight.data.copy_(pretrained_vit.encoder.ln.weight.data)
        self.norm.bias.data.copy_(pretrained_vit.encoder.ln.bias.data)

        # Classification head is NOT transferred -- different num_classes

    def _interpolate_pos_embed(self, pretrained_pos: torch.Tensor):
        """
        Interpolate pretrained positional embeddings to match our image size.

        When using a different image_size than the pretrained model (224),
        the number of patches changes. We interpolate the spatial positional
        embeddings using bicubic interpolation, keeping the CLS token embedding
        unchanged.
        """
        # Separate CLS token and patch embeddings
        cls_pos = pretrained_pos[:, :1, :]  # (1, 1, D)
        patch_pos = pretrained_pos[:, 1:, :]  # (1, N_pretrained, D)

        # Pretrained was 224/16 = 14x14 = 196 patches
        pretrained_grid = int(patch_pos.shape[1] ** 0.5)
        our_grid = self.image_size // self.patch_size

        # Reshape to 2D grid for interpolation
        patch_pos = patch_pos.reshape(1, pretrained_grid, pretrained_grid, -1)
        patch_pos = patch_pos.permute(0, 3, 1, 2)  # (1, D, H, W)

        # Bicubic interpolation to our grid size
        patch_pos = F.interpolate(
            patch_pos.float(),
            size=(our_grid, our_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        # Concatenate CLS token and interpolated patch embeddings
        self.pos_embed.data.copy_(torch.cat([cls_pos, patch_pos], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input tensor of shape (B, 3, H, W)

        Returns:
            logits of shape (B, num_classes)
        """
        B = x.shape[0]

        # Patch embedding: (B, 3, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use CLS token representation for classification
        cls_output = x[:, 0]  # (B, D)
        logits = self.head(cls_output)  # (B, num_classes)

        return logits

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from the last transformer block.

        This requires use_flash_attention=False because FlashAttention
        never materializes the attention matrix. If FlashAttention is
        enabled, this method temporarily switches to standard attention
        for the forward pass.

        Attention maps show WHAT the model attends to. For histopathology,
        we expect high attention on cell nuclei and tissue boundaries, not
        background.

        Args:
            x: input tensor of shape (B, 3, H, W)

        Returns:
            attention weights from the last block: (B, num_heads, N+1, N+1)
            where N = num_patches
        """
        # Temporarily disable flash attention on last block to get weights
        last_block = self.blocks[-1]
        original_flash = last_block.attn.use_flash
        last_block.attn.use_flash = False

        self.eval()
        with torch.no_grad():
            B = x.shape[0]

            # Patch embedding
            patches = self.patch_embed(x)
            patches = patches.flatten(2).transpose(1, 2)

            # Prepend CLS token and add positional embeddings
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, patches], dim=1)
            tokens = tokens + self.pos_embed
            tokens = self.pos_drop(tokens)

            # Pass through all blocks
            for block in self.blocks:
                tokens = block(tokens)

            # Get attention weights from last block
            attn_weights = last_block.attn.get_attention_weights()

        # Restore flash attention setting
        last_block.attn.use_flash = original_flash

        return attn_weights

    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters broken down by component.

        Returns:
            dict with keys: total, trainable, patch_embed, transformer,
                            classifier_head
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        patch_embed_params = sum(p.numel() for p in self.patch_embed.parameters())
        # CLS token + positional embeddings
        embed_params = self.cls_token.numel() + self.pos_embed.numel()

        transformer_params = sum(
            p.numel() for block in self.blocks for p in block.parameters()
        )
        norm_params = sum(p.numel() for p in self.norm.parameters())

        head_params = sum(p.numel() for p in self.head.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "patch_embed": patch_embed_params,
            "embeddings": embed_params,
            "transformer_blocks": transformer_params,
            "layer_norm": norm_params,
            "classifier_head": head_params,
        }


def measure_attention_memory(
    seq_len: int, num_heads: int = 12, head_dim: int = 64
) -> dict:
    """
    Computes theoretical memory usage for standard vs flash attention.

    Standard attention stores the full n x n attention score matrix in HBM.
    FlashAttention tiles the computation in SRAM and never writes n x n to HBM.

    Standard attention memory breakdown:
        - QKT matrix: n x n per head, stored in HBM for backward pass
        - Total: n x n x num_heads x 4 bytes (float32) per layer

    FlashAttention memory breakdown:
        - Only stores Q, K, V tensors (no n x n matrix)
        - Total: 3 x n x head_dim x num_heads x 4 bytes (float32)
        - The attention is computed in SRAM tiles and never written to HBM

    Args:
        seq_len: sequence length (num_patches + 1 for CLS token)
        num_heads: number of attention heads (default 12 for ViT-Base)
        head_dim: dimension per head (default 64, since 768/12 = 64)

    Returns:
        dict with keys:
            standard_attention_mb: memory for QKT matrix in MB (float32)
            flash_attention_mb: memory for Q, K, V blocks in MB (float32)
            reduction_factor: how many times smaller FlashAttention is
    """
    bytes_per_element = 4  # float32

    # Standard attention: stores the full n x n attention score matrix
    # Shape per head: (seq_len, seq_len), total across heads:
    standard_bytes = seq_len * seq_len * num_heads * bytes_per_element
    standard_mb = standard_bytes / (1024 * 1024)

    # FlashAttention: only stores Q, K, V (no n x n matrix in HBM)
    # Each of Q, K, V: (seq_len, head_dim) per head
    flash_bytes = 3 * seq_len * head_dim * num_heads * bytes_per_element
    flash_mb = flash_bytes / (1024 * 1024)

    reduction_factor = standard_mb / flash_mb if flash_mb > 0 else float("inf")

    return {
        "standard_attention_mb": round(standard_mb, 4),
        "flash_attention_mb": round(flash_mb, 4),
        "reduction_factor": round(reduction_factor, 2),
    }
