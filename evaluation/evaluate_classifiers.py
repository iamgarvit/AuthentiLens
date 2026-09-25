import os
import glob
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0
from PIL import Image
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from authentilens.paths import CHECKPOINTS, DATA_DIR, RESULTS_DIR, is_lfs_pointer

SD2_FR_TEST_DIR = DATA_DIR / "sd2-fr-testing"
SD2_CLASS_TEST_DIR = DATA_DIR / "sd2-classification" / "test"
OUTPUT_PATH = RESULTS_DIR / "classification" / "comprehensive_evaluation_results.json"

def create_resnet(device):
    model = resnet50(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 2))
    return model.to(device)

def create_efficientnet(device):
    model = efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    return model.to(device)

def load_checkpoint(model, path, device):
    state_dict = torch.load(path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '')
        cleaned_state_dict[clean_k] = v
        
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model

def evaluate_images(model, image_paths, true_label, transform, device, desc):
    correct_confs = []
    incorrect_confs = []
    
    with torch.no_grad():
        for path in tqdm(image_paths, desc=desc, leave=False):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred = probs.argmax().item()
                conf = probs[pred].item()
                
                if pred == true_label:
                    correct_confs.append(conf)
                else:
                    incorrect_confs.append(conf)
            except Exception as e:
                continue
                
    total = len(correct_confs) + len(incorrect_confs)
    acc = len(correct_confs) / total if total > 0 else 0
    return acc, total, len(correct_confs), len(incorrect_confs), correct_confs, incorrect_confs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Standard validation transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    models_config = {
        "ResNet_CIFake": (CHECKPOINTS["resnet50_cifake"], "resnet"),
        "ResNet_SD_1e-4": (CHECKPOINTS["resnet50_balanced_lr1e-4"], "resnet"),
        "ResNet_SD_2.5e-5": (CHECKPOINTS["resnet50_balanced_lr2.5e-5"], "resnet"),
        "EfficientNet_CIFake": (CHECKPOINTS["efficientnet_b0_cifake"], "efficientnet"),
        "EfficientNet_SD_1e-4": (CHECKPOINTS["efficientnet_b0_balanced_lr1e-4"], "efficientnet"),
        "EfficientNet_SD_2.5e-5": (CHECKPOINTS["efficientnet_b0_balanced_lr2.5e-5"], "efficientnet"),
    }

    # Data
    sd2_fr_testing_paths = glob.glob(f"{SD2_FR_TEST_DIR}/**/*.png", recursive=True) + glob.glob(f"{SD2_FR_TEST_DIR}/**/*.jpg", recursive=True)
    sd2_class_fake_paths = glob.glob(f"{SD2_CLASS_TEST_DIR}/FAKE/**/*.png", recursive=True) + glob.glob(f"{SD2_CLASS_TEST_DIR}/FAKE/**/*.jpg", recursive=True)
    sd2_class_real_paths = glob.glob(f"{SD2_CLASS_TEST_DIR}/REAL/**/*.png", recursive=True) + glob.glob(f"{SD2_CLASS_TEST_DIR}/REAL/**/*.jpg", recursive=True)

    print(f"Found {len(sd2_fr_testing_paths)} images in sd2-fr-testing (all fake).")
    print(f"Found {len(sd2_class_fake_paths)} FAKE and {len(sd2_class_real_paths)} REAL in sd2-classification/test.")

    final_results = {}

    for model_name, (ckpt_path, arch) in models_config.items():
        if not os.path.exists(ckpt_path):
            print(f"Warning: {ckpt_path} not found. Skipping {model_name}.")
            continue
        if is_lfs_pointer(ckpt_path):
            print(f"Warning: {ckpt_path} is a Git LFS pointer (see \"Get the weights\" in the README). Skipping {model_name}.")
            continue
            
        print(f"\nEvaluating {model_name}...")
        
        if arch == "resnet":
            model = create_resnet(device)
        else:
            model = create_efficientnet(device)
            
        model = load_checkpoint(model, ckpt_path, device)
        
        # 1. sd2-fr-testing (All FAKE)
        sd2_acc, sd2_tot, sd2_corr, sd2_incorr, sd2_c_confs, sd2_i_confs = evaluate_images(
            model, sd2_fr_testing_paths, true_label=0, transform=transform, device=device, desc=f"{model_name} SD2-FR-Testing"
        )
        
        # 2. sd2-classification/test (FAKE)
        cl_f_acc, cl_f_tot, cl_f_corr, cl_f_incorr, cl_f_c_confs, cl_f_i_confs = evaluate_images(
            model, sd2_class_fake_paths, true_label=0, transform=transform, device=device, desc=f"{model_name} SD2-Class FAKE"
        )
        
        # 3. sd2-classification/test (REAL)
        cl_r_acc, cl_r_tot, cl_r_corr, cl_r_incorr, cl_r_c_confs, cl_r_i_confs = evaluate_images(
            model, sd2_class_real_paths, true_label=1, transform=transform, device=device, desc=f"{model_name} SD2-Class REAL"
        )
        
        all_corr_confs = cl_f_c_confs + cl_r_c_confs
        all_incorr_confs = cl_f_i_confs + cl_r_i_confs
        
        cl_acc = (cl_f_corr + cl_r_corr) / (cl_f_tot + cl_r_tot) if (cl_f_tot + cl_r_tot) > 0 else 0

        final_results[model_name] = {
            "sd2-fr-testing_evaluation": {
                "dataset_type": "Original SD2 Generated full FAKE images",
                "total_images": sd2_tot,
                "accuracy": sd2_acc,
                "mean_confidence_correct": float(np.mean(sd2_c_confs)) if sd2_c_confs else 0.0,
                "std_confidence_correct": float(np.std(sd2_c_confs)) if sd2_c_confs else 0.0,
                "mean_confidence_incorrect": float(np.mean(sd2_i_confs)) if sd2_i_confs else 0.0,
                "std_confidence_incorrect": float(np.std(sd2_i_confs)) if sd2_i_confs else 0.0,
            },
            "sd2-classification_test_evaluation": {
                "dataset_type": "Cropped balancing of FAKE vs REAL",
                "total_images": cl_f_tot + cl_r_tot,
                "overall_accuracy": cl_acc,
                "fake_accuracy": cl_f_acc,
                "real_accuracy": cl_r_acc,
                "mean_confidence_correct": float(np.mean(all_corr_confs)) if all_corr_confs else 0.0,
                "std_confidence_correct": float(np.std(all_corr_confs)) if all_corr_confs else 0.0,
                "mean_confidence_incorrect": float(np.mean(all_incorr_confs)) if all_incorr_confs else 0.0,
                "std_confidence_incorrect": float(np.std(all_incorr_confs)) if all_incorr_confs else 0.0,
            }
        }
        
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print("\n--- Summary of Results ---")
    print(json.dumps(final_results, indent=4))
    print(f"Done! Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
