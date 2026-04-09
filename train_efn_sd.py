import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'data_dir': './sd2-classification',
    'save_dir': './checkpoints_efficientnet_sd',
    'epochs': 15,
    'batch_size': 256, 
    'num_workers': 8, 
    'lr': 2.5e-5,
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
                # Ignore macOS hidden files starting with ._ or .
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and not fname.startswith('.'):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))
                
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
        
    def __len__(self):
        return len(self.samples)

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
    all_preds, all_labels, all_confs = [], [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            confs = probs[torch.arange(len(probs)), preds]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    all_preds, all_labels, all_confs = np.array(all_preds), np.array(all_labels), np.array(all_confs)
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    fake_idx = (all_labels == 0)
    real_idx = (all_labels == 1)
    fake_acc = accuracy_score(all_labels[fake_idx], all_preds[fake_idx]) if fake_idx.sum() > 0 else 0.0
    real_acc = accuracy_score(all_labels[real_idx], all_preds[real_idx]) if real_idx.sum() > 0 else 0.0
    
    correct_mask = (all_preds == all_labels)
    confs_corr = all_confs[correct_mask]
    confs_incorr = all_confs[~correct_mask]
    
    metrics = {
        'loss': running_loss / len(loader),
        'acc': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fake_acc': float(fake_acc),
        'real_acc': float(real_acc),
        'mean_conf_correct': float(np.mean(confs_corr)) if len(confs_corr) > 0 else 0.0,
        'std_conf_correct': float(np.std(confs_corr)) if len(confs_corr) > 0 else 0.0,
        'mean_conf_incorrect': float(np.mean(confs_incorr)) if len(confs_incorr) > 0 else 0.0,
        'std_conf_incorrect': float(np.std(confs_incorr)) if len(confs_incorr) > 0 else 0.0,
    }
    return metrics

def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = FastImageFolder(os.path.join(CONFIG['data_dir'], 'train'), transform=train_transform)
    val_dataset = FastImageFolder(os.path.join(CONFIG['data_dir'], 'val'), transform=test_transform)
    test_dataset = FastImageFolder(os.path.join(CONFIG['data_dir'], 'test'), transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=CONFIG['num_workers'])

    print(f"Dataset Split -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    print("Loading Pretrained EfficientNet-B0 (bypassing hash mismatch)...")
    model = efficientnet_b0(weights=None)
    url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, check_hash=False)
    model.load_state_dict(state_dict)
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
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
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Validation")
        scheduler.step()

        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Acc: {val_metrics['acc']*100:.2f}% | FAKE Acc: {val_metrics['fake_acc']*100:.2f}% | REAL Acc: {val_metrics['real_acc']*100:.2f}% | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_metrics = val_metrics
            
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print("  [*] Best Model Saved!")

    # Save final model
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, os.path.join(CONFIG['save_dir'], 'final_model.pth'))

    print("\n--- FINAL TEST SET EVALUATION ---")
    model.load_state_dict(torch.load(os.path.join(CONFIG['save_dir'], 'best_model.pth')))
    test_metrics = evaluate(model, test_loader, criterion, device, desc="Testing (Best Model)")

    final_results = {
        'best_validation_metrics': best_metrics,
        'test_metrics': test_metrics
    }

    print("\n" + "="*40)
    print("FINAL TEST RESULTS:")
    print(f"  Accuracy:  {test_metrics['acc']*100:.2f}%")
    print(f"  Fake Acc:  {test_metrics['fake_acc']*100:.2f}%")
    print(f"  Real Acc:  {test_metrics['real_acc']*100:.2f}%")
    print(f"  Conf (Corr):   {test_metrics['mean_conf_correct']:.4f} ± {test_metrics['std_conf_correct']:.4f}")
    print(f"  Conf (Incorr): {test_metrics['mean_conf_incorrect']:.4f} ± {test_metrics['std_conf_incorrect']:.4f}")
    print("="*40)

    metrics_path = os.path.join(CONFIG['save_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Training Complete. Metrics successfully saved to {metrics_path}")

if __name__ == "__main__":
    main()
