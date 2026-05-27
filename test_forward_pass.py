import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.enhanced_model import Enhanced_RRSIS_UOT

def test_forward():
    print("Initializing Model...")
    # Initialize the model on CPU for a quick shape test
    model = Enhanced_RRSIS_UOT(
        sam3_ckpt=None,
        image_size=504,
        lora_rank=4,
        lora_alpha=8.0,
        use_dynamic_lora=True,
        use_contrastive_loss=True,
        use_multiscale_ot=True,
        use_ohem_loss=True,
    )
    
    B = 2
    # Mock inputs
    images = torch.randn(B, 3, 504, 504)
    # The EnhancedRRSIS_UOT forward actually expects captions as strings, 
    # but the VETextEncoder handles strings. Let's mock the captions.
    captions = ["a small red car", "a very large agricultural field near the river"]
    
    print("Running Forward Pass...")
    try:
        with torch.no_grad():
            outputs = model(images, captions)
            
        print("Forward pass successful!")
        print(f"Output mask shape: {outputs['pred_masks'].shape}")
        return True
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        return False

if __name__ == "__main__":
    test_forward()
