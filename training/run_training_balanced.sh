#!/bin/bash
set -e

# Run from anywhere: scripts live next to this file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "Starting ResNet-50 Training on SD2 dataset..."
echo "============================================="
python "$SCRIPT_DIR/train_resnet_balanced.py"

echo "============================================="
echo "Starting EfficientNet-B0 Training on SD2 dataset..."
echo "============================================="
python "$SCRIPT_DIR/train_efficientnet_balanced.py"

echo "All training completed!"
