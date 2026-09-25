import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as T
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from authentilens.paths import DATA_DIR, CLASSIFICATION_CKPT_DIR

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'data_dir': str(DATA_DIR / 'cifake'),  # CIFAKE: train/ and test/ with FAKE/ and REAL/
    'save_dir': str(CLASSIFICATION_CKPT_DIR / 'resnet50_cifake'),
    'epochs': 15,
    'batch_size': 256, 
    'num_workers': 8, 
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'image_size': 224,
    'seed': 42,
    'use_amp': True, 
}

# Reproducibility
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])

# ==========================================
# DATASET & DATALOADER
# ==========================================
class FastImageFolder(Dataset):
    """Memory-efficient and fast dataset loader using os.listdir"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_dir = os.path.join(root, cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
                
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
        
    def __len__(self):
        return len(self.samples)

class CustomSubset(Dataset):
    """Wrapper to apply transforms specifically on a given subset"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.subset)

# Standard ImageNet normalization
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

test_transform = T.Compose([
    T.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

# ==========================================
# TRAINING FUNCTIONS
# ==========================================
def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return running_loss / len(loader), acc, precision, recall, f1

def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Data
    print("Loading datasets...")
    base_train_dataset = FastImageFolder(os.path.join(CONFIG['data_dir'], 'train'))
    
    train_size = int(0.9 * len(base_train_dataset))
    val_size = len(base_train_dataset) - train_size
    train_subset, val_subset = random_split(base_train_dataset, [train_size, val_size])

    train_dataset = CustomSubset(train_subset, train_transform)
    val_dataset = CustomSubset(val_subset, test_transform)
    test_dataset = FastImageFolder(os.path.join(CONFIG['data_dir'], 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=CONFIG['num_workers'])

    print(f"Dataset Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Load Model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['use_amp'])

    best_val_acc = 0.0
    best_metrics = {}

    print(f"Starting training for {CONFIG['epochs']} epochs...")
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} Training")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=CONFIG['use_amp']): 
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}'})

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device, desc="Validation")
        scheduler.step()

        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                'epoch': epoch,
                'val_accuracy': val_acc, 'val_precision': val_prec, 
                'val_recall': val_rec, 'val_f1': val_f1
            }
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print("  --> Saved new best model!")

    # Save final model
    print("Training complete. Saving final model...")
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, os.path.join(CONFIG['save_dir'], 'final_model.pth'))

    # Evaluate final model on TEST SET
    print("\n--- FINAL TEST SET EVALUATION ---")
    # Load Best Model to evaluate on test set
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pth')))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device, desc="Testing (Best Model)")

    final_metrics = {
        'best_validation_metrics': best_metrics,
        'test_metrics': {
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_rec,
            'f1_score': test_f1
        }
    }

    print(f"Test Accuracy:  {test_acc*100:.2f}%")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")

    # Output to JSON
    metrics_path = os.path.join(CONFIG['save_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"Metrics successfully saved to {metrics_path}")

if __name__ == "__main__":
    main()