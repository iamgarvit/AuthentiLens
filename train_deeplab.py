import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from tqdm import tqdm
from PIL import Image
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'num_classes': 2,       # 0: Background/Real, 1: Fake/Manipulated Region
    'save_dir': './checkpoints_deeplab',
    'epochs': 15,
    'batch_size': 8,        # Segmentation uses more VRAM, keep batch size small
    'num_workers': 4,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'image_size': 256,      # DeepLab usually benefits from slightly larger images
    'seed': 42,
    'use_amp': True,        # Set to False if using older GPU like P100!
}

torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

# ==========================================
# DATASET DEFINITION
# ==========================================
class SegmentationDataset(Dataset):
    """
    Custom Dataset for Semantic Segmentation.
    You need to populate `self.image_paths` and `self.mask_paths`.
    """
    def __init__(self, mode="train"):
        self.mode = mode
        
        # ---------------------------------------------------------
        # TODO: DEFINE YOUR DATA LOADING LOGIC HERE
        # Example:
        # if mode == "train":
        #     self.image_paths = [...] 
        #     self.mask_paths = [...] 
        # ---------------------------------------------------------
        self.image_paths = [] 
        self.mask_paths = []  
        
        # Sanity check
        assert len(self.image_paths) == len(self.mask_paths), "Images and masks must match in length"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load Mask (Should be a 1-channel image where pixel values correspond to class indices, e.g., 0 and 1)
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert("L")
        
        # Apply deterministic transforms to BOTH image and mask identically
        img = img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.BILINEAR)
        mask = mask.resize((CONFIG['image_size'], CONFIG['image_size']), Image.NEAREST) # Nearest to keep discrete class values
        
        if self.mode == "train" and torch.rand(1).item() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Convert to Tensors
        img = TF.to_tensor(img)
        # Normalize image strictly matching ImageNet stats which DeepLab expects
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Convert mask to Long tensor
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3, self).__init__()
        # Matches your exact syntax from model_class.py using weights='DEFAULT'
        self.model = deeplabv3_resnet50(weights='DEFAULT')  
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
       
    def forward(self, x):
        return self.model(x)['out']

# ==========================================
# METRICS & EVALUATION
# ==========================================
def calculate_iou(preds, labels, num_classes):
    """Calculate Intersection over Union (IoU) for segmentation"""
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    # Ignore index 255 if using standard augmentations padding
    valid_mask = labels != 255
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in IoU
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    return np.nanmean(ious)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            with torch.cuda.amp.autocast(enabled=CONFIG['use_amp']):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_iou += calculate_iou(preds, masks, CONFIG['num_classes'])

    return running_loss / len(loader), total_iou / len(loader)

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Setting up datasets...")
    # NOTE: You MUST populate the placeholders in SegmentationDataset before running!
    # train_dataset = SegmentationDataset(mode="train")
    # val_dataset = SegmentationDataset(mode="val")
    # test_dataset = SegmentationDataset(mode="test")
    # train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    # val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    print("Loading Pretrained DeepLabV3...")
    model = DeepLabV3(num_classes=CONFIG['num_classes']).to(device)

    criterion = nn.CrossEntropyLoss()  # Standard loss for multiclass segmentation
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['use_amp'])

    best_val_iou = 0.0
    best_metrics = {}

    print(f"Starting DeepLabV3 Fine-Tuning for {CONFIG['epochs']} epochs...")
    """
    # Uncomment following blocks when dataset logic is populated:
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} Training")
        
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=CONFIG['use_amp']):
                out = model(images)
                loss = criterion(out, masks)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}'})

        train_loss = running_loss / len(train_loader)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_metrics = {'epoch': epoch, 'val_iou': val_iou, 'val_loss': val_loss}
            torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_deeplab.pth'))
            print("  --> Saved new best DeepLab checkpont!")

    print("Training complete. Saving final model...")
    torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'final_deeplab.pth'))

    # Final Test
    # test_loss, test_iou = evaluate(model, test_loader, criterion, device)
    # print(f"Final Test mIoU: {test_iou:.4f}")
    
    # Save Metrics File
    # metrics_path = os.path.join(CONFIG['save_dir'], 'metrics_deeplab.json')
    # with open(metrics_path, 'w') as f:
    #     json.dump({'best_val': best_metrics, 'test_iou': test_iou}, f, indent=4)
    """
    
    print("Script skeleton ready. Populate the dataset logic to begin training!")

if __name__ == "__main__":
    main()