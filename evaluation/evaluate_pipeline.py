import os
import glob
import json
import time
import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet50, efficientnet_b0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SEG_SIZE = 512
CLS_SIZE = 224

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
    deeplab.load_state_dict(state, strict=True)
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

def load_resnet(path):
    resnet = resnet50(pretrained=False)
    resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(resnet.fc.in_features, 2))
    state = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in state: state = state['model_state_dict']
    cache = {k.replace('module.', ''): v for k, v in state.items()}
    resnet.load_state_dict(cache)
    resnet.to(DEVICE)
    resnet.eval()
    return resnet

def load_efficientnet(path):
    effnet = efficientnet_b0(pretrained=False)
    num_ftrs = effnet.classifier[1].in_features
    effnet.classifier[1] = nn.Linear(num_ftrs, 2)
    state = torch.load(path, map_location=DEVICE)
    if 'model_state_dict' in state: state = state['model_state_dict']
    cache = {k.replace('module.', ''): v for k, v in state.items()}
    effnet.load_state_dict(cache)
    effnet.to(DEVICE)
    effnet.eval()
    return effnet

def run_eval(pixel_threshold):
    classifiers = {
        'ResNet': load_resnet('checkpoints_resnet_sd_2.5e-5/best_model.pth'),
        'EfficientNet': load_efficientnet('checkpoints_efficientnet_sd_2.5e-5/best_model.pth')
    }
    segmenters = {
        'DeepLabV3Plus': load_deeplab('checkpoints_deeplab/best_model.pth'),
        'UNet': load_unet('checkpoints_unet/best_model.pth')
    }
    
    img_paths = glob.glob('sd2-fr-testing/*/*.png')
    
    combinations = [
        ('ResNet', 'DeepLabV3Plus'),
        ('ResNet', 'UNet'),
        ('EfficientNet', 'DeepLabV3Plus'),
        ('EfficientNet', 'UNet')
    ]
    
    results = {combo[0] + "_" + combo[1]: {"classifier_correct": 0, "total_images": 0, "seg_iou": [], "seg_pixel_acc": [], "pipeline_fake_detected": 0} for combo in combinations}
    
    for img_path in tqdm(img_paths, desc="Processing images"):
        try:
            pil_image = Image.open(img_path).convert('RGB')
        except:
            continue
            
        cat = os.path.basename(os.path.dirname(img_path))
        fname = os.path.basename(img_path)
        base = fname.split('_sd2')[0]
        mask_path = os.path.join('sd2-fr-testing-masks', cat, base.replace('.png', '_512.png'))
        
        has_mask = os.path.exists(mask_path)
        if has_mask:
            gt_mask = Image.open(mask_path).convert('L').resize((SEG_SIZE, SEG_SIZE), Image.NEAREST)
            gt_mask = np.array(gt_mask) > 127
            
        # Get Classifier Predictions
        img_c = pil_image.resize((CLS_SIZE, CLS_SIZE), Image.BILINEAR)
        t_c = TF.to_tensor(img_c)
        t_c = TF.normalize(t_c, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)
        
        c_preds = {}
        with torch.no_grad():
            for cname, cmodel in classifiers.items():
                out = cmodel(t_c)
                pred = torch.softmax(out, dim=1)[0].argmax().item()
                c_preds[cname] = pred  # 0 is FAKE, 1 is REAL
                
        # Get Seg Predictions
        img_s = pil_image.resize((SEG_SIZE, SEG_SIZE), Image.BILINEAR)
        t_s = TF.to_tensor(img_s)
        t_s = TF.normalize(t_s, mean=MEAN, std=STD).unsqueeze(0).to(DEVICE)
        
        s_preds = {}
        with torch.no_grad():
            for sname, smodel in segmenters.items():
                out = smodel(t_s)
                prob = torch.sigmoid(out)[0, 0].cpu().numpy()
                bin_map = (prob > pixel_threshold)
                iou = 0.0
                pixel_acc = 0.0
                if has_mask:
                    intersection = np.logical_and(bin_map, gt_mask).sum()
                    union = np.logical_or(bin_map, gt_mask).sum()
                    iou = intersection / union if union > 0 else (1.0 if not gt_mask.any() and not bin_map.any() else 0.0)
                    pixel_acc = (bin_map == gt_mask).mean()
                flagged_pct = float(bin_map.mean()) * 100
                s_preds[sname] = {"iou": iou, "pixel_acc": pixel_acc, "flagged_pct": flagged_pct}
                
        # Combinations logic
        for cname in classifiers.keys():
            for sname in segmenters.keys():
                combo_key = f"{cname}_{sname}"
                res = results[combo_key]
                res["total_images"] += 1
                
                is_cls_fake = (c_preds[cname] == 0)
                if is_cls_fake:
                    res["classifier_correct"] += 1
                
                if has_mask:
                    res["seg_iou"].append(s_preds[sname]["iou"])
                    res["seg_pixel_acc"].append(s_preds[sname]["pixel_acc"])
                
                # Pipeline logic: if the user's heuristic says FAKE via Seg pct > 30% when class != FAKE
                is_seg_fake = s_preds[sname]["flagged_pct"] > 30.0
                if is_cls_fake or is_seg_fake:
                    res["pipeline_fake_detected"] += 1

    # Summarize
    summary = {}
    for k, v in results.items():
        summary[k] = {
            "total_images": v["total_images"],
            "classifier_fake_TPR": v["classifier_correct"] / v["total_images"] if v["total_images"] > 0 else 0,
            "segmentation_mIoU": float(np.mean(v["seg_iou"])) if v["seg_iou"] else 0.0,
            "segmentation_pixel_accuracy": float(np.mean(v["seg_pixel_acc"])) if v["seg_pixel_acc"] else 0.0,
            "combined_pipeline_TPR": v["pipeline_fake_detected"] / v["total_images"] if v["total_images"] > 0 else 0
        }
        
    with open(f'combinations_test_results_{pixel_threshold}.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved combinations_test_results_{pixel_threshold}.json")

    print(json.dumps(summary, indent=4))

if __name__ == '__main__':
    for t in [0.6, 0.7]:
        print(f"Evaluating with segmentation pixel threshold {t}...")
        run_eval(t)