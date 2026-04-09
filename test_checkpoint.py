#!/usr/bin/env python3
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

print("✓ Libraries imported successfully")

# Match the updated backend architecture
class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=1, decoder_dropout=0.3):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        
        if decoder_dropout > 0:
            self.model.decoder.block = nn.Sequential(
                nn.Dropout2d(p=decoder_dropout),
                self.model.decoder.block2,
            )
       
    def forward(self, x):
        return self.model(x)

# Create model with updated architecture
model = CustomDeepLabV3(num_classes=1, decoder_dropout=0.3)
print("✓ Model architecture created")

# Test checkpoint files in priority order
checkpoints = [
    'checkpoints_deeplab/best_deeplabv3plus (1).pth',  # Working version (highest priority)
    'checkpoints_deeplab/best_model.pth',
    'checkpoints_deeplab/best_deeplabv3plus.pth'
]

success = False
for ckpt_path in checkpoints:
    print(f"\n{'='*60}")
    print(f"Testing: {ckpt_path}")
    print(f"{'='*60}")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"✓ Checkpoint file opened successfully")
        print(f"  Keys in checkpoint: {list(ckpt.keys())}")
        
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
            print(f"  model_state_dict has {len(state)} tensors")
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
            print(f"  state_dict has {len(state)} tensors")
        else:
            state = ckpt
            print(f"  Direct state dict with {len(state)} tensors")
        
        # Try loading into model
        model.model.load_state_dict(state)
        print(f"✓✓✓ Weights loaded into model successfully!")
        success = True
        break
        
    except FileNotFoundError:
        print(f"✗ File not found: {ckpt_path}")
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)[:200]}")

print(f"\n{'='*60}")
if success:
    print("SUCCESS! Checkpoint file is valid and ready to use.")
else:
    print("FAILED: No valid checkpoint found.")
print(f"{'='*60}\n")

