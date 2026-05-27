import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingAwarePromptGenerator(nn.Module):
    """
    Grounding-Aware Prompts Generation (GPG) module inspired by SGSRF.

    This module takes the OT transport plan (which acts as a structural-semantic
    alignment map) and generates explicit geometric prompts for the SAM3 Decoder.
    
    It extracts:
    1. Sparse Prompts (Points): Coordinates of the highest probability regions.
    2. Dense Prompts (Masks): Low-resolution coarse mask priors.
    """
    def __init__(self, num_points=1):
        super().__init__()
        self.num_points = num_points
        print(f"[GPG] Initialized Grounding-Aware Prompt Generator (num_points={num_points})")

    @torch.no_grad()
    def forward(self, P, text_mask, original_image_size, device):
        """
        Generate geometric prompts from the OT transport plan.

        Args:
            P: (B, HW, seq) transport plan from the deepest OT aligner.
            text_mask: (B, seq) boolean mask (True = padding).
            original_image_size: int, size of the input image (e.g., 504).
            device: torch.device.

        Returns:
            points: (B, num_points, 2) normalized coordinates [0, 1] or absolute?
                    SAM3 expects absolute coordinates if scaled, but for the FindStage 
                    we can provide absolute pixel coordinates.
            points_mask: (B, num_points) boolean mask.
            dense_mask: (B, 1, 256, 256) dense mask prior for SAM3.
        """
        B, HW, seq = P.shape
        H = W = int(HW ** 0.5)

        # 1. Mask out padding tokens in the transport plan
        if text_mask is not None:
            # text_mask is True for padding. We want to keep valid tokens (False)
            valid_mask = (~text_mask).unsqueeze(1)  # (B, 1, seq)
            P_valid = P * valid_mask.float()
        else:
            P_valid = P

        # 2. Aggregate over sequence to get a global spatial heatmap
        heatmap = P_valid.sum(dim=-1)  # (B, HW)
        heatmap = heatmap.view(B, H, W)

        # 3. Scale-Aware Prompting (SAP) from ReSaP & Sparse Prompts Generation
        flat_heatmap = heatmap.view(B, -1)
        
        # Measure target area ratio for scale awareness
        b_max = flat_heatmap.max(dim=1, keepdim=True)[0] + 1e-6
        norm_heatmap = flat_heatmap / b_max
        active_area = (norm_heatmap > 0.5).sum(dim=1).float() / (H * W)
        
        # Extract max 5 points for structure coverage
        K_max = 5
        _, topk_idx = torch.topk(flat_heatmap, K_max, dim=-1)

        # Convert 1D indices to 2D (y, x) coordinates in the feature map space
        y_feat = torch.div(topk_idx, W, rounding_mode='floor')
        x_feat = topk_idx % W

        # Scale coordinates to the original image size
        scale_y = original_image_size / H
        scale_x = original_image_size / W

        y_img = (y_feat.float() + 0.5) * scale_y
        x_img = (x_feat.float() + 0.5) * scale_x

        # SAM3 expects points as (x, y)
        points = torch.stack((x_img, y_img), dim=-1)  # (B, K_max, 2)
        
        # Initialize masks and labels (1 = foreground point, -1 = padding/ignore)
        points_mask = torch.ones((B, K_max), dtype=torch.bool, device=device)
        point_labels = torch.ones((B, K_max), dtype=torch.long, device=device)
        
        # Apply dynamic scale-adaptive logic per image in the batch
        for b in range(B):
            if active_area[b] < 0.01:
                # Tiny object: Keep only 1 center point to avoid background noise
                points_mask[b, 1:] = False
                point_labels[b, 1:] = -1

        # We intentionally omit the dense_mask because SAM3's Image Predictor GeometryEncoder 
        # is not pre-trained with a MaskEncoder. Passing it would introduce untrained parameters.
        return points, points_mask, point_labels
