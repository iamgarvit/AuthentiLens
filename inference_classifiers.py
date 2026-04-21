import torch
import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b0
import torchvision.transforms as T
from PIL import Image
import sys
import os

# ==========================================
# CONFIGURATION
# ==========================================
RESNET_CHECKPOINT = "./checkpoints_resnet/best_model.pth"
EFFICIENTNET_CHECKPOINT = "./checkpoints_efficientnet/best_model.pth"

# Change these based on your archive's folder names!
# FastImageFolder sorted them alphabetically, so look at your train folder alphabetically:
# E.g., if folders are "FAKE" and "REAL", FAKE=0, REAL=1
CLASS_NAMES = {0: "Class 0 (e.g. FAKE)", 1: "Class 1 (e.g. REAL)"} 

IMAGE_SIZE = 224

# Setup device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Standard ImageNet normalization used during training
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# MODEL SETUP FUNCTIONS
# ==========================================
def load_resnet(checkpoint_path, device):
    print("Loading ResNet-50...")
    model = resnet50(pretrained=False)
    # Recreate the exact head used in training
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    if os.path.exists(checkpoint_path):
        # map_location ensures we can load server-trained GPU models locally on Mac (MPS/CPU)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    else:
        print(f"  [!] ResNet checkpoint not found at {checkpoint_path}")
        return None

def load_efficientnet(checkpoint_path, device):
    print("Loading EfficientNet-B0...")
    model = efficientnet_b0(pretrained=False)
    # Recreate the exact head used in training
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    else:
        print(f"  [!] EfficientNet checkpoint not found at {checkpoint_path}")
        return None

# ==========================================
# INFERENCE LOGIC
# ==========================================
def predict(model, image_tensor, device, model_name="Model"):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        
        pred_class = CLASS_NAMES.get(predicted_idx.item(), f"Class {predicted_idx.item()}")
        
        print(f"[{model_name}] Prediction: {pred_class} | Confidence: {confidence.item()*100:.2f}%")
        # Print breakdown for transparency
        print(f"      -> {CLASS_NAMES[0]}: {probabilities[0].item()*100:.2f}%")
        print(f"      -> {CLASS_NAMES[1]}: {probabilities[1].item()*100:.2f}%")

def main(image_path):
    print(f"\n--- Running Inference on Device: {device} ---")
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)
        
    print(f"Processing Image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

    # Load and run ResNet
    resnet_model = load_resnet(RESNET_CHECKPOINT, device)
    if resnet_model:
        predict(resnet_model, image_tensor, device, model_name="ResNet-50")
        
    print("-" * 40)
    
    # Load and run EfficientNet
    eff_model = load_efficientnet(EFFICIENTNET_CHECKPOINT, device)
    if eff_model:
        predict(eff_model, image_tensor, device, model_name="EfficientNet-B0")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_classifiers.py <path_to_image>")
    else:
        main(sys.argv[1])
