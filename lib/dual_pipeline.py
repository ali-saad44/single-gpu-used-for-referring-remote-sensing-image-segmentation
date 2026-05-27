import torch
import torch.nn as nn
import torch.nn.functional as F
from .enhanced_model import Enhanced_RRSIS_UOT

class DualPipeline_RRSIS_UOT(nn.Module):
    """
    Dual-Stage referring remote sensing image segmentation pipeline.
    Stage 1: Coarse localization at 504x504.
    Stage 2: High-resolution boundary refinement at 800x800 using Stage 1's predictions as guidance.
    """
    def __init__(
        self,
        sam3_ckpt,
        lora_rank=16,
        lora_alpha=32.0,
        freeze_backbone=True,
        freeze_text_encoder=True,
        gradient_checkpointing=True,
        # Shared/Stage 1 enhancement options
        use_dynamic_lora=True,
        use_contrastive_loss=True,
        use_multiscale_ot=True,
        use_ohem_loss=True,
        contrastive_weight=0.1,
        ohem_hard_ratio=0.3,
        ot_reg=0.1,
        ot_num_iter=10,
        num_ot_scales=3,
        **kwargs
    ):
        super().__init__()
        
        # Build Stage 1 (resolution = 504)
        print("Initializing Stage 1 Model (504x504)...")
        self.stage1_model = Enhanced_RRSIS_UOT(
            sam3_ckpt=sam3_ckpt,
            image_size=504,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_backbone=freeze_backbone,
            freeze_text_encoder=freeze_text_encoder,
            gradient_checkpointing=gradient_checkpointing,
            use_dynamic_lora=use_dynamic_lora,
            use_contrastive_loss=use_contrastive_loss,
            use_multiscale_ot=use_multiscale_ot,
            use_ohem_loss=use_ohem_loss,
            contrastive_weight=contrastive_weight,
            ohem_hard_ratio=ohem_hard_ratio,
            ot_reg=ot_reg,
            ot_num_iter=ot_num_iter,
            num_ot_scales=num_ot_scales
        ).to('cpu')
        
        # Build Stage 2 (resolution = 800)
        print("Initializing Stage 2 Model (800x800)...")
        self.stage2_model = Enhanced_RRSIS_UOT(
            sam3_ckpt=sam3_ckpt,
            image_size=800,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_backbone=freeze_backbone,
            freeze_text_encoder=freeze_text_encoder,
            gradient_checkpointing=gradient_checkpointing,
            use_dynamic_lora=use_dynamic_lora,
            use_contrastive_loss=use_contrastive_loss,
            use_multiscale_ot=use_multiscale_ot,
            use_ohem_loss=use_ohem_loss,
            contrastive_weight=contrastive_weight,
            ohem_hard_ratio=ohem_hard_ratio,
            ot_reg=ot_reg,
            ot_num_iter=ot_num_iter,
            num_ot_scales=num_ot_scales
        ).to('cpu')
        for param in self.stage1_model.parameters():
            param.requires_grad = False
            
    def load_stage1_weights(self, path, device='cpu'):
        """Load pretrained checkpoint weights into Stage 1 and freeze it."""
        print(f"Loading Stage 1 weights from {path}...")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Load weights
        self.stage1_model.load_state_dict(state_dict, strict=False)
        print("Stage 1 weights loaded successfully.")
        
        # Ensure Stage 1 is in eval mode and frozen
        self.stage1_model.eval()
        for param in self.stage1_model.parameters():
            param.requires_grad = False

    def initialize_stage2_from_stage1(self):
        """Initialize Stage 2 weights from Stage 1 to bootstrap training."""
        print("Initializing Stage 2 weights from Stage 1...")
        state_dict = self.stage1_model.state_dict()
        self.stage2_model.load_state_dict(state_dict, strict=False)
        print("Stage 2 weights initialized from Stage 1.")

    def forward(self, images, captions, masks_gt=None):
        B = images.shape[0]
        device = images.device

        # Stage 1 operates on 504x504. Interpolate images down to 504.
        images_504 = F.interpolate(
            images.float(),
            size=(504, 504),
            mode='bilinear',
            align_corners=False
        )

        # 1. Run Stage 1 (without gradients)
        self.stage1_model.eval()
        with torch.no_grad():
            stage1_out = self.stage1_model(images_504, captions)
            
        pred_masks_504 = stage1_out['pred_masks'] # [B, 1, 504, 504] logits

        # 2. Transition Logic: Scale-Aware Prompting (SAP)
        stage2_points_list = []
        stage2_labels_list = []
        stage2_masks_list = []
        
        K_max = 25 # Maximum prompt points (e.g. 5x5 grid)
        
        for b in range(B):
            mask_504 = pred_masks_504[b, 0] # [504, 504]
            # Convert logits to binary mask using 0.0 threshold (equivalent to sigmoid > 0.5)
            binary_mask = (mask_504 > 0.0)
            
            # Check if mask is empty to prevent crashes
            if not binary_mask.any():
                # FALLBACK: If mask is empty, sample center of the image
                pt = torch.tensor([[400.0, 400.0]], device=device) # center of 800x800
                
                # Pad to K_max
                pts = torch.zeros(K_max, 2, device=device)
                pts[0] = pt
                labels = torch.ones(K_max, device=device, dtype=torch.long) # default to 1 (foreground)
                
                # PyTorch convention: True for padded/ignored tokens, False for active/valid tokens
                mask_val = torch.ones(K_max, device=device, dtype=torch.bool)
                mask_val[0] = False # Only first point is active
            else:
                # Find indices of all foreground pixels
                y_indices, x_indices = torch.where(binary_mask)
                
                # Get bounding box coordinates in 504x504 space
                ymin, ymax = y_indices.min().float(), y_indices.max().float()
                xmin, xmax = x_indices.min().float(), x_indices.max().float()
                
                # Calculate scale of the object (bbox size) in 504 space
                h_box = ymax - ymin + 1.0
                w_box = xmax - xmin + 1.0
                diag_box = torch.sqrt(h_box**2 + w_box**2)
                
                # Determine point density grid size based on scale
                if diag_box < 50.0:
                    grid_size = 1
                elif diag_box < 150.0:
                    grid_size = 3 # 3x3 grid (9 points)
                else:
                    grid_size = 5 # 5x5 grid (25 points)
                    
                # Compute centroid or grid points in 504 space
                sampled_pts_504 = []
                if grid_size == 1:
                    # Centroid
                    cy = y_indices.float().mean()
                    cx = x_indices.float().mean()
                    sampled_pts_504.append((cx, cy))
                else:
                    # Grid points within bounding box
                    y_steps = torch.linspace(ymin + h_box*0.1, ymax - h_box*0.1, grid_size, device=device)
                    x_steps = torch.linspace(xmin + w_box*0.1, xmax - w_box*0.1, grid_size, device=device)
                    for gy in y_steps:
                        for gx in x_steps:
                            sampled_pts_504.append((gx, gy))
                            
                # Scale coordinates to 800x800 space (multiply by 800/504 = 1.5873)
                scale_factor = 800.0 / 504.0
                pts = torch.zeros(K_max, 2, device=device)
                labels = torch.ones(K_max, device=device, dtype=torch.long)
                mask_val = torch.ones(K_max, device=device, dtype=torch.bool)
                
                for idx, (cx, cy) in enumerate(sampled_pts_504):
                    if idx >= K_max:
                        break
                    cx_800 = cx * scale_factor
                    cy_800 = cy * scale_factor
                    
                    # Bound checking to prevent coordinate overflow
                    cx_800 = torch.clamp(cx_800, 0.0, 799.0)
                    cy_800 = torch.clamp(cy_800, 0.0, 799.0)
                    
                    pts[idx] = torch.stack([cx_800, cy_800])
                    labels[idx] = 1 # Foreground point
                    mask_val[idx] = False # False means active/valid in PyTorch Transformer key_padding_mask
                    
            stage2_points_list.append(pts)
            stage2_labels_list.append(labels)
            stage2_masks_list.append(mask_val)

        # Collate points into tensors
        stage2_points = torch.stack(stage2_points_list, dim=0) # [B, K_max, 2]
        stage2_point_labels = torch.stack(stage2_labels_list, dim=0) # [B, K_max]
        stage2_points_mask = torch.stack(stage2_masks_list, dim=0) # [B, K_max]

        # Convert Stage 1 logits mask to dense probability map (sigmoid)
        prev_mask_pred = torch.sigmoid(pred_masks_504) # [B, 1, 504, 504]

        # 3. Run Stage 2 at 800x800 with the prompt guidance
        stage2_out = self.stage2_model(
            images=images,
            captions=captions,
            masks_gt=masks_gt,
            stage1_points=stage2_points,
            stage1_points_mask=stage2_points_mask,
            stage1_point_labels=stage2_point_labels,
            prev_mask_pred=prev_mask_pred
        )
        
        # Include stage 1 outputs for downstream analysis/visualization
        stage2_out['stage1_pred_masks'] = pred_masks_504
        
        return stage2_out
