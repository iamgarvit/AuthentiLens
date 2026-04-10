# Presentation Outline: Deepfake & Inpainting Detection

*(Note: Slides 1 & 2 covering Problem Statement and Related Works are omitted as per instructions.)*

---

Slide 3: Dataset and Evaluation Metrics

Datasets Used
CIFAKE: Used initially for baseline classification training.
SD2 FR: A subset of the TGIF2 dataset containing original SD2-generated full FAKE images. Primarily used to train our segmentation models (UNet and DeepLabV3Plus).
Custom Classification Dataset: A balanced dataset derived directly from the SD2 FR dataset explicitly to train and evaluate our classification models robustly.

Evaluation Metrics
Classification: Overall Accuracy, Fake Accuracy (Accuracy on Fake images), Real Accuracy (accuracy on real images).
Segmentation: mean Intersection over Union (mIoU), Pixel Accuracy.
Pipeline: Combined True Positive Rate (TPR) for the joint heuristic classification.

Slide 4: System Baselines and Initial Failure Case

Segmentation Baselines
Trained UNet and DeepLabV3Plus architectures directly on the SD2 FR dataset to predict manipulated pixel masks.

Classification Baselines and The Suspicious Success
Initially trained ResNet-50 and EfficientNet-B0 on the standard CIFAKE dataset.
The Failure Case: When evaluated against the SD2 FR testing set, the models performed suspiciously well (e.g., ResNet achieved 100% accuracy).
The Catch: The original SD2 FR testing set consisted entirely of FAKE images.
The Revelation: When we evaluated these CIFAKE-trained models against our newly created balanced Custom Classification test set, performance collapsed entirely. ResNet had a 1.7% Real Accuracy (it was just blindly predicting FAKE for almost every image).

The Need for a Custom Dataset
The original SD2 FR dataset only contained FAKE images:
Train: 7320 FAKE Images
Validation: 1023 FAKE Images
Test: 1029 FAKE Images
Because it lacked REAL images, any classifier could achieve 100% accuracy simply by becoming heavily biased to always predict FAKE. 

Custom Dataset Methodology
To fix this, we created a balanced classification dataset directly from the SD2 FR source by extracting 224x224 crops from the original 512x512 images:
For FAKE images: We dynamically extracted a 224x224 crop that strictly contained the masked/manipulated region.
For REAL images: We scanned the same 512x512 images to find a 224x224 region that had absolutely zero overlap with the manipulation mask (representing pristine, unedited pixels).
We only retained images that could successfully fulfill both conditions. This yielded a perfectly balanced Real vs Fake pairing for our models to learn from.

Custom Dataset Breakdown
Train Split: 5567 Real / 5567 Fake (11134 total)
Validation Split: 798 Real / 798 Fake (1596 total)
Test Split: 816 Real / 816 Fake (1632 total)

Slide 5: Custom Dataset Methodology and Rectification

Methodological Fix
We trained our ResNet and EfficientNet models directly on this new Custom Classification Dataset.
The models successfully unlearned the always fake bias and produced more balanced, reliable predictions across both Real and Fake samples.

Table 1: Classification Results Analysis

Model / Training Dataset / Evaluation Dataset / Overall Acc / Fake Acc / Real Acc
ResNet / CIFAKE / SD2 FR (All Fakes) / 100.0% / 100.0% / N/A
EfficientNet / CIFAKE / SD2 FR (All Fakes) / 68.2% / 68.2% / N/A
ResNet / CIFAKE / Custom (Balanced) / 50.4% / 99.1% / 1.7%
EfficientNet / CIFAKE / Custom (Balanced) / 50.0% / 67.4% / 32.6%
ResNet / Custom / Custom (Balanced) / 63.1% / 61.0% / 65.2%
EfficientNet / Custom / Custom (Balanced) / 61.3% / 50.9% / 71.7%

Slide 6: Error Rectification via Heuristics

Overcoming Classification Limits
While training on the Custom Dataset fixed the bias, base classification accuracy still hovered around 61 to 63 percent.
Solution: We introduced a joint heuristic pipeline. If a classifier fails to detect a fake, but our Segmentation models flag a sufficiently large manipulated region (e.g., >30% of pixels at a 0.7 probability threshold), the system overrides the classifier and flags the image as FAKE.

Table 2: Combined Pipeline Results (Segmentation Threshold = 0.7)

Pipeline Combination / Classifier Fake TPR / Segmentation mIoU / Seg Pixel Acc / Combined Pipeline TPR
ResNet + DeepLabV3Plus / 59.0% / 16.8% / 61.8% / 94.8%
ResNet + UNet / 59.0% / 17.3% / 63.3% / 86.6%
EfficientNet + DeepLabV3Plus / 71.4% / 16.8% / 61.8% / 96.7%
EfficientNet + UNet / 71.4% / 17.3% / 63.3% / 91.7%

Conclusion: System pipeline TPR improves massively by allowing segmentation heatmaps to act as a fallback error-rectifier.

Slide 7: Next Steps and Summary

Future Work and Direction
1. Joint Training: Instead of training and running classification and segmentation separately (which yields lower standalone classification accuracy), we plan to investigate end-to-end joint training to allow the networks to share spatial and contextual features.
2. Frequency and Noise Domain Features: Taking inspiration from State-Of-The-Art models like DIRE and FIRE, we intend to introduce frequency domain or noise-residual features into our models, rather than relying solely on RGB pixel data.
3. Enhanced Web Interface: For the final presentation, our live evaluation platform will be updated to allow users to toggle between current models, our upcoming joint-training models, and established SOTA baseline models for side-by-side performance comparisons.

Contributions:


Slide 8: Demo

Live demonstration of the AuthentiLens Streamlit web application.
Showcasing simultaneous multi-model inference, visualization of segmentation probability masks vs binary masks, and real-time system performance profiling.