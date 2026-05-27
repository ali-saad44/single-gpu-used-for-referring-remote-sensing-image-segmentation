"""
Enhanced_RRSIS_UOT: Enhanced model wrapping RRSIS_SAM3 with 4 new techniques.

Enhancements over RRSIS_SAM3:
    1. Text-Guided Dynamic LoRA — text-conditioned vision adapter weights
    2. Contrastive Language-Image Loss — InfoNCE alignment supervision
    3. Multi-Scale OT Feature Alignment — per-FPN-level OT alignment
    4. OHEM Loss — hard pixel mining + focal + boundary-aware supervision

All 4 techniques can be individually enabled/disabled via flags for
ablation studies.

Architecture:
    Input (Image + Text)
        → SAM3 VL Backbone (ViT + Text Encoder)
            + Dynamic LoRA (text-conditioned vision adaptation) [NEW]
        → Multi-Scale OT Alignment (per-FPN-level) [ENHANCED]
        → Transformer Encoder (fusion)
        → DETR Decoder (detection)
        → Segmentation Head (mask prediction)
        → OHEM + Focal + Boundary Loss [ENHANCED]
        → Contrastive Loss (auxiliary) [NEW]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sam3.model_builder import (
    build_sam3_image_model,
    _create_text_encoder,
    _create_vision_backbone,
    _create_vl_backbone,
    _create_sam3_transformer,
    _create_dot_product_scoring,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_sam3_model,
    _load_checkpoint,
)
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt

from .rs_adapters import inject_lora_adapters, get_trainable_params_summary
from .dynamic_lora import inject_dynamic_lora_adapters, DynamicLoRAManager
from .multiscale_ot_alignment import MultiScaleOTAligner
from .contrastive_loss import ContrastiveLoss
from .ohem_loss import EnhancedOHEMLoss
from .ot_feature_alignment import OTFeatureAligner
from .ot_loss import OTSegmentationLoss
from .prompt_generator import GroundingAwarePromptGenerator


class Enhanced_RRSIS_UOT(nn.Module):
    """
    Enhanced RRSIS model with Unbalanced Optimal Transport and
    4 novel techniques for improved performance.

    This model wraps SAM3 and extends the base RRSIS_SAM3 with:
    - Text-Guided Dynamic LoRA for text-aware vision adaptation
    - Multi-Scale OT alignment across all FPN levels
    - Contrastive loss for vision-language alignment
    - OHEM loss for handling extreme class imbalance

    All enhancements are individually toggleable for ablation.

    Args:
        sam3_ckpt: Path to SAM3 pretrained checkpoint.
        image_size: Input image size (divisible by 14).
        lora_rank: LoRA rank for adapters.
        lora_alpha: LoRA scaling factor.
        freeze_backbone: Whether to freeze the ViT backbone.
        freeze_text_encoder: Whether to freeze the text encoder.
        gradient_checkpointing: Enable gradient checkpointing.
        use_dynamic_lora: Enable text-guided dynamic LoRA.
        use_contrastive_loss: Enable contrastive loss.
        use_multiscale_ot: Enable multi-scale OT alignment.
        use_ohem_loss: Enable OHEM + Focal + Boundary loss.
        contrastive_weight: Weight for contrastive loss.
        ohem_hard_ratio: OHEM hard pixel ratio.
        ot_reg: OT Sinkhorn regularization.
        ot_num_iter: Number of Sinkhorn iterations.
        num_ot_scales: Number of FPN scales for multi-scale OT.
    """

    def __init__(
        self,
        sam3_ckpt: str = None,
        image_size: int = 504,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_backbone: bool = True,
        freeze_text_encoder: bool = True,
        gradient_checkpointing: bool = True,
        # === Enhancement flags ===
        use_dynamic_lora: bool = True,
        use_contrastive_loss: bool = True,
        use_multiscale_ot: bool = True,
        use_ohem_loss: bool = True,
        # === Enhancement params ===
        contrastive_weight: float = 0.1,
        scl_weight: float = 0.1,
        ohem_hard_ratio: float = 0.3,
        ot_reg: float = 0.1,
        ot_num_iter: int = 10,
        num_ot_scales: int = 3,
    ):
        super().__init__()
        self.image_size = image_size
        self.use_dynamic_lora = use_dynamic_lora
        self.use_contrastive_loss = use_contrastive_loss
        self.use_multiscale_ot = use_multiscale_ot
        self.use_ohem_loss = use_ohem_loss
        self.contrastive_weight = contrastive_weight
        self.scl_weight = scl_weight

        # ====== Build SAM3 Image Model ======
        print("[Enhanced_RRSIS_UOT] Building SAM3 image model...")
        is_random = (sam3_ckpt == "random")
        actual_ckpt = None if is_random else sam3_ckpt
        self.sam3 = build_sam3_image_model(
            device="cpu",
            eval_mode=False,
            checkpoint_path=actual_ckpt,
            load_from_HF=False if is_random else (actual_ckpt is None),
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )

        # ====== Freeze Strategy ======
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_text_encoder:
            self._freeze_text_encoder()

        # ====== LoRA Injection ======
        d_model = self.sam3.hidden_dim  # typically 256

        if use_dynamic_lora:
            print("[Enhanced_RRSIS_UOT] Injecting Text-Guided Dynamic LoRA...")
            _, self.lora_manager = inject_dynamic_lora_adapters(
                self.sam3, text_dim=d_model, rank=lora_rank, alpha=lora_alpha
            )
        else:
            print("[Enhanced_RRSIS_UOT] Using standard static LoRA...")
            inject_lora_adapters(self.sam3, rank=lora_rank, alpha=lora_alpha)
            self.lora_manager = None

        # ====== Unfreeze trainable components ======
        self._unfreeze_trainable_components()

        # ====== Gradient Checkpointing ======
        if gradient_checkpointing:
            print("[Enhanced_RRSIS_UOT] Gradient checkpointing enabled")

        # ====== Multi-Scale OT Alignment ======
        if use_multiscale_ot:
            print(f"[Enhanced_RRSIS_UOT] Multi-Scale OT Alignment ({num_ot_scales} scales)")
            self.ms_ot_aligner = MultiScaleOTAligner(
                d_model=d_model,
                num_scales=num_ot_scales,
                reg=ot_reg,
                num_iter=ot_num_iter,
            )
        else:
            # Fallback to single-scale OT from RRSIS_SAM3
            print("[Enhanced_RRSIS_UOT] Single-Scale OT Alignment (baseline)")
            self.ot_aligner = OTFeatureAligner(
                d_model=d_model,
                reg=ot_reg,
                num_iter=ot_num_iter,
            )

        # ====== Contrastive Loss ======
        if use_contrastive_loss:
            print("[Enhanced_RRSIS_UOT] Contrastive Loss (InfoNCE) enabled")
            self.contrastive_loss = ContrastiveLoss(
                visual_dim=d_model,
                text_dim=d_model,
            )

        # ====== OHEM Loss ======
        if use_ohem_loss:
            print("[Enhanced_RRSIS_UOT] OHEM + FocalDice + Boundary Loss enabled")
            self.enhanced_loss = EnhancedOHEMLoss(hard_ratio=ohem_hard_ratio)
        else:
            print("[Enhanced_RRSIS_UOT] Standard Dice+BCE Loss (baseline)")
            self.standard_loss = OTSegmentationLoss()

        # ====== Grounding-Aware Prompt Generator ======
        print("[Enhanced_RRSIS_UOT] Grounding-Aware Prompt Generation (GPG) enabled")
        self.gpg = GroundingAwarePromptGenerator(num_points=1)

        # ====== Mask Projector for Dense Prior Guidance ======
        self.mask_projector = nn.Conv2d(1, d_model, kernel_size=1)

        # ====== Print Summary ======
        get_trainable_params_summary(self)

        # Image normalization
        self.register_buffer('pixel_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    # ================================================================
    # Freeze / Unfreeze helpers
    # ================================================================

    def _freeze_backbone(self):
        """Freeze the ViT vision backbone."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'vision_backbone'):
            for param in backbone.vision_backbone.parameters():
                param.requires_grad = False
            print("[Enhanced_RRSIS_UOT] ViT backbone frozen")

    def _freeze_text_encoder(self):
        """Freeze the text encoder."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'language_backbone'):
            for param in backbone.language_backbone.parameters():
                param.requires_grad = False
            print("[Enhanced_RRSIS_UOT] Text encoder frozen")

    def _unfreeze_trainable_components(self):
        """Unfreeze components we want to fine-tune."""
        if hasattr(self.sam3, 'transformer'):
            for param in self.sam3.transformer.encoder.parameters():
                param.requires_grad = True
            for param in self.sam3.transformer.decoder.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Transformer encoder+decoder unfrozen")

        if self.sam3.segmentation_head is not None:
            for param in self.sam3.segmentation_head.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Segmentation head unfrozen")

        if hasattr(self.sam3, 'geometry_encoder'):
            for param in self.sam3.geometry_encoder.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Geometry encoder unfrozen")

        if hasattr(self.sam3, 'dot_prod_scoring') and self.sam3.dot_prod_scoring is not None:
            for param in self.sam3.dot_prod_scoring.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Scoring head unfrozen")

    def normalize_image(self, images):
        """Normalize images to SAM3's expected range [-1, 1]."""
        return (images - self.pixel_mean) / self.pixel_std

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(self, images, captions, masks_gt=None, stage1_points=None, stage1_points_mask=None, stage1_point_labels=None, prev_mask_pred=None):
        """
        Enhanced forward pass with all 4 techniques.

        Args:
            images: [B, 3, H, W] tensor of RS images.
            captions: List[str] of length B, referring text descriptions.
            masks_gt: [B, 1, H, W] ground truth masks (for loss).
            stage1_points: Optional [B, K, 2] external point prompts.
            stage1_points_mask: Optional [B, K] point prompt masks.
            stage1_point_labels: Optional [B, K] point prompt labels.
            prev_mask_pred: Optional [B, 1, H_mask, W_mask] dense mask prior from stage 1.

        Returns:
            dict with 'pred_masks', 'pred_logits', 'pred_boxes', 'loss'.
        """
        B = images.shape[0]
        device = images.device

        # Normalize images
        images = self.normalize_image(images)

        # ====== Step 1: Forward Text First ======
        text_out = self.sam3.backbone.forward_text(captions, device=device)

        # ====== Step 1.5: Dynamic LoRA text conditioning ======
        if self.use_dynamic_lora and self.lora_manager is not None:
            text_feats = text_out.get('language_features', None)
            if text_feats is not None:
                # Pool text features for conditioning: (seq, B, C) → (B, C)
                if text_feats.dim() == 3:
                    pooled_text = text_feats.mean(dim=0)  # (B, C)
                else:
                    pooled_text = text_feats
                self.lora_manager.set_text_conditioning(pooled_text)

        # ====== Step 1.8: Forward Image (Now with Text Context!) ======
        backbone_out = self.sam3.backbone.forward_image(images)
        backbone_out.update(text_out)

        # Fix: Ensure vision_pos_enc is on the correct device
        # Positional encodings may be cached on CPU if the model was initialized with device="cpu"
        if "vision_pos_enc" in backbone_out and backbone_out["vision_pos_enc"] is not None:
            backbone_out["vision_pos_enc"] = [
                pos.to(device) if isinstance(pos, torch.Tensor) else pos 
                for pos in backbone_out["vision_pos_enc"]
            ]
        
        if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
            if "vision_pos_enc" in backbone_out["sam2_backbone_out"]:
                backbone_out["sam2_backbone_out"]["vision_pos_enc"] = [
                    pos.to(device) if isinstance(pos, torch.Tensor) else pos 
                    for pos in backbone_out["sam2_backbone_out"]["vision_pos_enc"]
                ]

        # ====== Step 2: OT Feature Alignment ======
        text_feats = backbone_out.get('language_features', None)
        text_mask = backbone_out.get('language_mask', None)

        if text_feats is not None and 'backbone_fpn' in backbone_out:
            fpn_feats = backbone_out['backbone_fpn']

            if self.use_multiscale_ot and hasattr(self, 'ms_ot_aligner'):
                # Multi-Scale OT alignment (NEW)
                aligned_fpn = self.ms_ot_aligner(fpn_feats, text_feats, text_mask)
                backbone_out['backbone_fpn'] = aligned_fpn
            elif hasattr(self, 'ot_aligner'):
                # Single-scale OT alignment (baseline fallback)
                for i in range(len(fpn_feats)):
                    feat = fpn_feats[i]
                    if hasattr(feat, 'tensors'):
                        feat_tensor = feat.tensors
                    else:
                        feat_tensor = feat
                    if feat_tensor.dim() == 4:
                        aligned = self.ot_aligner(feat_tensor, text_feats, text_mask)
                        if hasattr(feat, 'tensors'):
                            feat.tensors = aligned
                        else:
                            fpn_feats[i] = aligned

        # ====== Step 3: Prompt Generation via GPG Module ======
        img_ids = torch.arange(B, device=device)
        text_ids = torch.arange(B, device=device)
        
        # Get the transport plan from the last OT aligner
        P_map = None
        if self.use_multiscale_ot and hasattr(self, 'ms_ot_aligner'):
            last_aligner = self.ms_ot_aligner.scale_aligners[-1]
            P_map = getattr(last_aligner, 'last_P', None)
        elif hasattr(self, 'ot_aligner'):
            P_map = getattr(self.ot_aligner, 'last_P', None)
            
        input_points = torch.zeros(B, 0, 2, device=device)
        input_points_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
        input_point_labels = torch.zeros(B, 0, device=device, dtype=torch.long)
        
        if stage1_points is not None:
            input_points = stage1_points
            input_points_mask = stage1_points_mask
            input_point_labels = stage1_point_labels
        elif P_map is not None:
            # Generate explicit geometric prompts from structural alignment with Scale-Aware Prompting
            input_points, input_points_mask, input_point_labels = self.gpg(P_map, text_mask, self.image_size, device)

        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=torch.zeros(B, 0, 4, device=device),
            input_boxes_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
            input_boxes_label=torch.zeros(B, 0, device=device, dtype=torch.long),
            input_points=input_points,
            input_points_mask=input_points_mask,
        )

        has_geom = (input_points.numel() > 0 or (stage1_points is not None and stage1_points.size(1) > 0))

        if has_geom:
            # point_embeddings: [N_points, B, 2] - Must be normalized to [0, 1] for SAM3 pos encoder
            point_embeddings = (input_points / self.image_size).transpose(0, 1)
            # point_labels: [N_points, B]
            point_labels = input_point_labels.transpose(0, 1)
            
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, B, 4, device=device),
                box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
                point_embeddings=point_embeddings,
                point_mask=input_points_mask,
                point_labels=point_labels,
            )
        else:
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, B, 4, device=device),
                box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
            )

        # ====== Step 4: Encode Prompt ======
        formatted_prev_mask = None
        if prev_mask_pred is not None:
            # backbone_out["backbone_fpn"] has shape (B, C, H_feat, W_feat)
            if "backbone_fpn" in backbone_out and len(backbone_out["backbone_fpn"]) > 0:
                feat = backbone_out["backbone_fpn"][-1]
                if hasattr(feat, "tensors"):
                    feat_tensor = feat.tensors
                else:
                    feat_tensor = feat
                H_feat, W_feat = feat_tensor.shape[-2:]
                
                # Downsample prev_mask_pred to (H_feat, W_feat)
                # prev_mask_pred: [B, 1, 504, 504]
                downsampled_mask = F.interpolate(
                    prev_mask_pred.float(),
                    size=(H_feat, W_feat),
                    mode='bilinear',
                    align_corners=False
                ) # shape: [B, 1, H_feat, W_feat]
                
                # Project 1 channel to d_model channels
                projected_mask = self.mask_projector(downsampled_mask) # shape: [B, d_model, H_feat, W_feat]
                
                # Convert shape to [H_feat * W_feat, B, d_model] to match img_feats[-1]
                formatted_prev_mask = projected_mask.flatten(2).permute(2, 0, 1)

        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt,
            prev_mask_pred=formatted_prev_mask
        )

        # ====== Step 5: Run Encoder (fusion) ======
        backbone_out, encoder_out, feat_tuple = self.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

        # ====== Step 6: Run Decoder (DETR detection) ======
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        }
        out, hs = self.sam3._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        # ====== Step 7: Segmentation Head ======
        if self.sam3.segmentation_head is not None:
            _, _, _, vis_feat_sizes = feat_tuple
            seg_img_ids = find_input.img_ids
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                seg_img_ids = backbone_out["id_mapping"][seg_img_ids]

            self.sam3._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=seg_img_ids,
                vis_feat_sizes=vis_feat_sizes,
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        # ====== Step 8: Select Best Mask ======
        result = self._select_best_mask(out, B)

        # ====== Step 9: Clear Dynamic LoRA conditioning ======
        # if self.use_dynamic_lora and self.lora_manager is not None:
        #     self.lora_manager.clear_text_conditioning()

        # ====== Step 10: Compute Losses ======
        if masks_gt is not None:
            # Primary segmentation loss
            if self.use_ohem_loss and hasattr(self, 'enhanced_loss'):
                seg_loss = self.enhanced_loss(result, masks_gt, self.image_size)
            elif hasattr(self, 'standard_loss'):
                seg_loss = self.standard_loss(result, masks_gt, self.image_size)
            else:
                seg_loss = self._compute_fallback_loss(result['pred_masks'], masks_gt)

            # SCL Loss (Structural Consistency Loss)
            scl_loss = torch.tensor(0.0, device=device)
            if self.use_multiscale_ot and hasattr(self, 'ms_ot_aligner'):
                scl_vals = [a.scl_loss for a in self.ms_ot_aligner.scale_aligners if isinstance(getattr(a, 'scl_loss', 0.0), torch.Tensor)]
                if scl_vals:
                    scl_loss = sum(scl_vals) / len(scl_vals)

            # Contrastive loss (auxiliary)
            contrastive = torch.tensor(0.0, device=device)
            if self.use_contrastive_loss and hasattr(self, 'contrastive_loss'):
                try:
                    # Use FPN features with real spatial structure (not pooled encoder output)
                    fpn_feats = backbone_out.get('backbone_fpn', None)
                    if fpn_feats is not None and text_feats is not None and len(fpn_feats) > 0:
                        # Use highest-resolution FPN level for spatial contrastive learning
                        vis_feat = fpn_feats[0]
                        if hasattr(vis_feat, 'tensors'):
                            vis_feat = vis_feat.tensors
                        if vis_feat.dim() == 4:  # (B, C, H, W)
                            contrastive = self.contrastive_loss(
                                vis_feat, text_feats,
                                result['pred_masks'], masks_gt
                            )
                except Exception as e:
                    # Don't crash training if contrastive loss fails
                    if self.training:
                        print(f"[WARNING] Contrastive loss skipped: {e}")
                    contrastive = torch.tensor(0.0, device=device)

            result['loss'] = seg_loss + self.contrastive_weight * contrastive + self.scl_weight * scl_loss
            result['seg_loss'] = seg_loss.detach()
            result['scl_loss'] = scl_loss.detach() if isinstance(scl_loss, torch.Tensor) else scl_loss
            result['contrastive_loss'] = contrastive.detach() if isinstance(contrastive, torch.Tensor) else contrastive

        return result

    def _select_best_mask(self, out, batch_size):
        """Select the best mask from SAM3's multi-query output."""
        result = {}

        pred_logits = out.get('pred_logits', None)
        if pred_logits is not None:
            result['pred_logits'] = pred_logits

        pred_boxes = out.get('pred_boxes', None)
        if pred_boxes is not None:
            result['pred_boxes'] = pred_boxes

        pred_masks = out.get('pred_masks', None)
        if pred_masks is not None:
            # Store all query masks for IoU-based score supervision
            result['all_query_masks'] = pred_masks  # (B, N, H_mask, W_mask)

            if pred_logits is not None:
                scores = pred_logits.squeeze(-1)
                best_idx = scores.argmax(dim=-1)
                batch_idx = torch.arange(batch_size, device=pred_masks.device)
                best_masks = pred_masks[batch_idx, best_idx]
                best_masks = best_masks.unsqueeze(1)
            else:
                best_masks = pred_masks[:, 0:1]

            best_masks = F.interpolate(
                best_masks.float(),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
            result['pred_masks'] = best_masks
        else:
            result['pred_masks'] = torch.zeros(
                batch_size, 1, self.image_size, self.image_size,
                device=next(self.parameters()).device,
            )

        return result

    def _compute_fallback_loss(self, pred_masks, gt_masks):
        """Fallback Dice + BCE loss."""
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(), size=pred_masks.shape[-2:], mode='nearest'
            )

        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())

        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        return 0.5 * bce_loss + 0.5 * dice_loss

    @torch.no_grad()
    def predict(self, images, captions):
        """Inference-only forward pass."""
        self.eval()
        result = self.forward(images, captions)
        result['pred_probs'] = torch.sigmoid(result['pred_masks'])
        result['pred_binary'] = (result['pred_probs'] > 0.5).float()
        return result
