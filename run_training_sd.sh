#!/bin/bash
set -e

echo "============================================="
echo "Starting ResNet-50 Training on SD2 dataset..."
echo "============================================="
python train_resnet_sd.py

echo "============================================="
echo "Starting EfficientNet-B0 Training on SD2 dataset..."
echo "============================================="
python train_efn_sd.py

echo "All training completed!"
