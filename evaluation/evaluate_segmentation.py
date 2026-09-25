import os
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEG_SIZE = 512

def load_deeplab(path):
    deeplab = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    deeplab.decoder.block2 = nn.Sequential(
        deeplab.decoder.block2,
        nn.Dropout2d(p=0.3),
    )
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)

    is_buggy = any(k.startswith('decoder.block.') for k in state.keys())
    if is_buggy:
        new_state = {}
        for k, v in state.items():
            if k.startswith('decoder.block.1.'):
                suffix = k[len('decoder.block.1.'):]
                new_key = f'decoder.block2.0.{suffix}'
                new_state[new_key] = v
            elif k.startswith('decoder.block.'):
                pass
            elif k.startswith('decoder.block2.'):
                pass
            else:
                new_state[k] = v
        state = new_state
    deeplab.load_state_dict(state, strict=False)
    deeplab.to(DEVICE)
    deeplab.eval()
    return deeplab

def load_unet(path):
    unet = smp.Unet(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
    )
    dropout_p = 0.3
    for i, block in enumerate(unet.decoder.blocks):
        n = len(unet.decoder.blocks)
        scaled = dropout_p * (0.4 + 0.6 * i / max(n - 1, 1))
        block.conv2 = nn.Sequential(block.conv2, nn.Dropout2d(p=scaled))
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    unet.load_state_dict(state, strict=True)
    unet.to(DEVICE)
    unet.eval()
    return unet

def run_seg_eval(pixel_threshold=0.7):
    segmenters = {
        'DeepLabV3Plus': load_deeplab('checkpoints_deeplab/best_model.pth'),
        'UNet': load_unet('checkpoints_unet/best_model.pth')
    }
    
    img_paths = glob.glob('sd2-fr-testing/*/*.png')
    
    results = {
        'DeepLabV3Plus': {"iou": [], "pixel_acc": []},
        'UNet': {"iou": [], "pixel_acc": []}
    }
    
    for img_path in tqdm(img_paths, desc="Processing images"):
        try:
            pil_image = Image.open(img_path).convert('RGB')
        except Exception:
            continue
            
        cat = os.path.basename(os.path.dirname(img_path))
        fname = os.path.basename(img_path)
        base = fname.split('_sd2')[0]
        mask_path = os.path.join('sd2-fr-testing-masks', cat, base.replace('.png', '_512.png'))
        
        has_mask = os.path.exists(mask_path)
        if not has_mask:
            continue # Only evaluate when we have ground truth masks
            
        gt_mask = Image.open(mask_path).convert('L').resize((SEG_SIZE, SEG_SIZE), Image.NEAREST)
        gt_mask = np.array(gt_mask) > 127
            
        img_s = pil_image.resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
        t_s = TF.to_tensor(img_s)
        t_s = TF.normalize(t_s, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            for sname, smodel in segmenters.items():
                out = smodel(t_s)
                prob = torch.sigmoid(out)[0, 0].cpu().numpy()
                bin_map = (prob > pixel_threshold)
                
                intersection = np.logical_and(bin_map, gt_mask).sum()
                union = np.logical_or(bin_map, gt_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                else:
                    iou = 1.0 if not gt_mask.any() and not bin_map.any() else 0.0
                    
                pixel_acc = (bin_map == gt_mask).mean()
                
                results[sname]["iou"].append(iou)
                results[sname]["pixel_acc"].append(pixel_acc)
                
    summary = {}
    for sname, metrics in results.items():
        summary[sname] = {
            "mIoU": float(np.mean(metrics["iou"])) if len(metrics["iou"]) > 0 else 0.0,
            "pixel_accuracy": float(np.mean(metrics["pixel_acc"])) if len(metrics["pixel_acc"]) > 0 else 0.0,
            "total_evaluated": len(metrics["iou"])
        }
        
    print(f"--- Segmentation Metrics (Threshold: {pixel_threshold}) ---")
    print(json.dumps(summary, indent=4))
    
    with open('segmentation_only_results.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("Saved to segmentation_only_results.json")

if __name__ == '__main__':
    run_seg_eval(0.7)
