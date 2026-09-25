import json
import os
from pathlib import Path

def check_accuracy(registry_file, votes_file):
    with open(registry_file, 'r') as f:
        registry = json.load(f)
        
    with open(votes_file, 'r') as f:
        votes = json.load(f)
        
    correct = 0
    total = 0
    tied = 0
    
    unique_voters = set()
    
    # confusion matrix
    true_real = 0
    true_fake = 0
    pred_real_true_real = 0
    pred_fake_true_fake = 0
    
    for img_id, v_data in votes.items():
        for vote_record in v_data.get("votes", []):
            unique_voters.add(vote_record.get("session_id"))
            
        if img_id not in registry:
            continue
            
        gt = registry[img_id]["ground_truth"]
        real_votes = v_data.get("total_real_votes", 0)
        fake_votes = v_data.get("total_fake_votes", 0)
        
        if real_votes == 0 and fake_votes == 0:
            continue # no votes
            
        total += 1
        
        if gt == "REAL":
            true_real += 1
        else:
            true_fake += 1
            
        if real_votes > fake_votes:
            pred = "REAL"
        elif fake_votes > real_votes:
            pred = "FAKE"
        else:
            pred = "TIE"
            tied += 1
            
        if pred == gt:
            correct += 1
            if gt == "REAL":
                pred_real_true_real += 1
            else:
                pred_fake_true_fake += 1
                
    accuracy = correct / total if total > 0 else 0
    
    print(f"Results for {os.path.basename(votes_file)}:")
    print(f"  Total Evaluated: {total}")
    print(f"  Unique Voters (Sessions): {len(unique_voters)}")
    print(f"  Correct: {correct}")
    print(f"  Tied: {tied}")
    print(f"  Accuracy: {accuracy:.4f}")
    if true_real > 0:
        print(f"  REAL class accuracy: {pred_real_true_real}/{true_real} ({pred_real_true_real/true_real:.4f})")
    if true_fake > 0:
        print(f"  FAKE class accuracy: {pred_fake_true_fake}/{true_fake} ({pred_fake_true_fake/true_fake:.4f})")
    print()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    
    print("--- CUSTOM DATASET ---")
    check_accuracy(f"{base_dir}/data/custom_registry.json", f"{base_dir}/results/custom_votes.json")
    
    print("--- SD2-FR DATASET ---")
    check_accuracy(f"{base_dir}/data/sd2_fr_registry.json", f"{base_dir}/results/sd2_fr_votes.json")
