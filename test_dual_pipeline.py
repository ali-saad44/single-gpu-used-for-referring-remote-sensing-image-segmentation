import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.dual_pipeline import DualPipeline_RRSIS_UOT

def verify_dual_pipeline():
    print("=" * 60)
    print("1. Initializing Dual Pipeline in offline/random mode...")
    print("=" * 60)
    
    # Instantiate with sam3_ckpt="random" to run offline without downloading weights
    model = DualPipeline_RRSIS_UOT(
        sam3_ckpt="random",
        lora_rank=4,
        lora_alpha=8.0,
        freeze_backbone=True,
        freeze_text_encoder=True,
        gradient_checkpointing=False,  # Disable gradient checkpointing for CPU testing
        use_dynamic_lora=True,
        use_contrastive_loss=True,
        use_multiscale_ot=True,
        use_ohem_loss=True,
    )
    
    print("\nModel created successfully.")
    
    # 2. Prepare mock inputs
    print("\n2. Preparing mock inputs...")
    B = 2
    images = torch.randn(B, 3, 800, 800)
    captions = [
        "a small building in the corner",
        "dense green forest area next to the circular highway"
    ]
    masks_gt = torch.randint(0, 2, (B, 1, 800, 800)).float()
    
    print(f"Input images shape: {images.shape}")
    print(f"Input captions: {captions}")
    print(f"Input masks_gt shape: {masks_gt.shape}")
    
    # Set model to training mode
    model.train()
    
    # Ensure Stage 1 is in eval mode and Stage 2 is in train mode
    model.stage1_model.eval()
    model.stage2_model.train()
    
    # 3. Forward Pass
    print("\n3. Running Forward Pass...")
    outputs = model(images, captions, masks_gt=masks_gt)
    
    print("Forward pass completed successfully!")
    print(f"Outputs keys: {list(outputs.keys())}")
    
    pred_masks = outputs['pred_masks']
    loss = outputs['loss']
    
    print(f"Output mask shape: {pred_masks.shape} (Expected: [2, 1, 800, 800])")
    print(f"Loss value: {loss.item()}")
    
    # Assertions for shapes
    assert pred_masks.shape == (B, 1, 800, 800), f"Incorrect mask shape: {pred_masks.shape}"
    assert loss.dim() == 0, f"Loss is not a scalar: {loss.shape}"
    
    # 4. Backward Pass
    print("\n4. Running Backward Pass...")
    loss.backward()
    print("Backward pass completed successfully!")
    
    # 5. Programmatic freeze verification
    print("\n5. Verifying frozen vs trainable parameters...")
    
    # Stage 1 Verification
    stage1_frozen = True
    stage1_total_params = 0
    for name, param in model.stage1_model.named_parameters():
        stage1_total_params += 1
        if param.requires_grad:
            print(f"WARNING: Stage 1 param '{name}' requires_grad=True!")
            stage1_frozen = False
        if param.grad is not None:
            print(f"WARNING: Stage 1 param '{name}' received gradients!")
            stage1_frozen = False
            
    if stage1_frozen:
        print(f"SUCCESS: All {stage1_total_params} Stage 1 parameters are frozen and received NO gradients.")
    else:
        print("FAILURE: Some Stage 1 parameters are not frozen or received gradients!")
        
    # Stage 2 Verification
    stage2_trainable_with_grad = 0
    stage2_trainable_no_grad = []
    
    for name, param in model.stage2_model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                stage2_trainable_with_grad += 1
            else:
                stage2_trainable_no_grad.append(name)
                
    print(f"Stage 2 trainable parameters with gradients: {stage2_trainable_with_grad}")
    if len(stage2_trainable_no_grad) > 0:
        print(f"Stage 2 trainable parameters without gradients: {len(stage2_trainable_no_grad)}")
        # Print a few of them for debugging
        for name in stage2_trainable_no_grad[:5]:
            print(f"  - {name}")
            
    # We should have some trainable parameters in Stage 2, and they should receive gradients.
    assert stage2_trainable_with_grad > 0, "No Stage 2 parameters received gradients!"
    assert stage1_frozen, "Stage 1 parameters were not properly frozen!"
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        verify_dual_pipeline()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
