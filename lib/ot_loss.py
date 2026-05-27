"""
OT-Enhanced Segmentation Loss for RRSIS_SAM3.

Uses Dice + BCE mask loss as the primary supervision signal.
The OT-based feature alignment (in ot_feature_alignment.py) provides
the main architectural contribution from Optimal Transport — spatial
text-to-image fusion via Sinkhorn matching.

This loss module focuses on robust, stable mask supervision that
complements the OT aligner upstream.

Reference:
    De Plaen et al., "Unbalanced Optimal Transport: A Unified Framework
    for Object Detection", CVPR 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTSegmentationLoss(nn.Module):
    """
    Segmentation loss for RRSIS with OT-enhanced feature alignment.

    Computes Dice + BCE loss on predicted mask logits. The OT contribution
    comes from the upstream OTFeatureAligner module which performs
    Sinkhorn-based spatial text-to-image fusion before the encoder.

    For referring segmentation (single target per image), direct mask
    supervision is more effective than detection-style box matching,
    since we always have exactly one target per query.

    Args:
        dice_weight: Weight for Dice loss component.
        bce_weight: Weight for BCE loss component.
        score_weight: Weight for query confidence loss (optional).
    """

    def __init__(
        self,
        dice_weight=5.0,
        bce_weight=2.0,
        score_weight=1.0,
        **kwargs,  # Accept and ignore unused OT kwargs for backward compat
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.score_weight = score_weight
        print("[OTLoss] Dice+BCE mask loss with OT feature alignment")

    def forward(self, outputs, gt_masks, image_size):
        """
        Compute the segmentation loss.

        Args:
            outputs: dict from RRSIS_SAM3 forward containing:
                - 'pred_masks': (B, 1, H, W) predicted mask logits
                - 'pred_logits': (B, N_queries, 1) confidence scores (optional)
            gt_masks: (B, 1, H, W) ground truth binary masks.
            image_size: int, spatial size of images.

        Returns:
            total_loss: scalar tensor.
        """
        # === 1. Mask Loss (Dice + BCE) ===
        mask_loss = self._compute_mask_loss(outputs['pred_masks'], gt_masks)

        # === 2. Score supervision (optional) ===
        score_loss = torch.tensor(0.0, device=gt_masks.device)
        if ('pred_logits' in outputs
                and outputs['pred_logits'] is not None
                and gt_masks is not None):
            score_loss = self._compute_score_loss(
                outputs['pred_logits'], gt_masks,
                outputs.get('all_query_masks', None)
            )

        # === 3. Total Loss ===
        total = mask_loss + self.score_weight * score_loss

        return total

    def _compute_mask_loss(self, pred_masks, gt_masks):
        """Dice + BCE loss on predicted mask logits."""
        # Resize GT to match prediction if needed
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_resized = F.interpolate(
                gt_masks.float(), pred_masks.shape[-2:], mode='nearest'
            )
        else:
            gt_resized = gt_masks.float()

        # Binary cross-entropy
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_resized)

        # Dice loss
        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_resized).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_resized.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice

    def _compute_score_loss(self, pred_logits, gt_masks, all_query_masks=None):
        """
        Supervise query confidence scores with IoU-based matching.

        The query whose mask best overlaps with GT gets target=1.0,
        others get target=0.0. This teaches the model to rank its own
        mask hypotheses by quality.
        """
        B = gt_masks.shape[0]
        device = gt_masks.device

        if pred_logits.dim() == 3:
            # pred_logits: (B, N_queries, 1)
            scores = pred_logits.squeeze(-1)  # (B, N)
            N = scores.shape[1]

            with torch.no_grad():
                has_object = (gt_masks.sum(dim=(1, 2, 3)) > 0).float()  # (B,)

                if all_query_masks is not None and has_object.sum() > 0:
                    # IoU-based matching: find which query best matches GT
                    qm = all_query_masks  # (B, N, H_mask, W_mask)
                    gt_for_iou = gt_masks.float()
                    if qm.shape[-2:] != gt_for_iou.shape[-2:]:
                        gt_for_iou = F.interpolate(
                            gt_for_iou, qm.shape[-2:], mode='nearest'
                        )
                    gt_expanded = gt_for_iou.squeeze(1).unsqueeze(1).expand_as(qm)
                    pred_binary = (torch.sigmoid(qm) > 0.5).float()
                    intersection = (pred_binary * gt_expanded).sum(dim=(2, 3))
                    union = pred_binary.sum(dim=(2, 3)) + gt_expanded.sum(dim=(2, 3)) - intersection
                    per_query_iou = (intersection + 1e-6) / (union + 1e-6)  # (B, N)

                    target_scores = torch.zeros_like(scores)
                    best_query = per_query_iou.argmax(dim=1)
                    target_scores[torch.arange(B, device=device), best_query] = 1.0
                    target_scores = target_scores * has_object.unsqueeze(1)
                else:
                    target_scores = torch.zeros_like(scores)

            loss = F.binary_cross_entropy_with_logits(
                scores, target_scores, reduction='mean'
            )
            return loss

        return torch.tensor(0.0, device=device)
