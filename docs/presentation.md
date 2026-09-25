# Presentation Outline: Deepfake & Inpainting Detection

Team name: Aperture
Project name: AuthentiLens - Visualizing AI Manipulations
People -
Divyanshu Yadav (2023211)
Garvit (2023217)
Rewant Anand (2023429)
Tanish Bachhas (2023545)

---
Slide 1:
Problem Statement:
The rapid advancement of generative AI models such as Stable Diffusion and Midjourney has enabled the creation of highly realistic synthetic images as well as subtle manipulations of real photographs through techniques like inpainting and outpainting. While existing detection systems primarily focus on classifying entire images as real or fake, they struggle to identify localized manipulations within otherwise authentic images. This creates a critical gap, as many real-world misuse cases involve partial edits rather than fully generated content.

Project Objective & Scope:
To address this limitation, our project aims to develop a web-based system capable of both classifying images as “Real” or “AI-generated” and producing pixel-wise heatmaps that localize manipulated regions. The scope of the project is restricted to detecting artifacts introduced by diffusion-based generative models in photorealistic scenes, explicitly excluding artistic or stylized images to maintain a focused and well-defined problem setting.

Users & Applications:
This system is motivated by real-world applications where image authenticity is critical. Potential users include food delivery platforms that need to detect fake food images used in refund scams, insurance companies verifying the authenticity of damage claims, social media platforms aiming to curb misinformation, and academic institutions that require validation of submitted documents. The common requirement across these domains is the ability to detect not just fake images, but subtle manipulations within real ones.

Slide 2: 
Paragraph 1 (SOTA Methods Overview):
Our work builds upon recent advances in synthetic image forensics, particularly methods designed to detect artifacts introduced by generative models. DIRE leverages diffusion reconstruction error to identify pixels that can be overly well reconstructed by generative models, indicating synthetic content. FIRE focuses on frequency-guided reconstruction to capture inconsistencies in mid-frequency image statistics, while “Art or Artifact?” shifts attention toward segmenting localized artifacts instead of relying purely on global classification.

Paragraph 2 (Key Methodological Critique):
A key observation across these state-of-the-art approaches is their reliance on frequency-domain or noise-domain cues, such as reconstruction error patterns or spectral inconsistencies, to detect synthetic content. While effective, these methods implicitly assume access to such auxiliary signals and may become less robust when these artifacts are weak, removed, or mimicked by real images. Moreover, they do not fully explore how far standard vision models can go using purely spatial and semantic information present in the image itself, without relying on handcrafted or domain-specific signals.

Paragraph 3 (Baseline Design Motivation):
Motivated by this gap, we design our baselines using a combination of classification and segmentation models that operate directly on image pixels. The goal is to evaluate how well conventional deep learning architectures, without explicit frequency or reconstruction-based features, can perform both global detection and localized forgery identification. This allows us to establish a strong and interpretable baseline, and to understand whether complex frequency-based methods are strictly necessary or if comparable performance can be achieved through end-to-end learning from visual data alone.

Slide 3: Dataset, Metrics, and Experimental Framing

Paragraph 1 (Datasets Used):
To support both classification and localization tasks, we adopt a combination of curated and custom datasets. For segmentation, we use SD2-FR, a subset of the TGIF2 dataset, which contains text-guided inpainting-based manipulations along with pixel-level annotations, making it well-suited for learning localized forgery detection. For classification, we constructed a custom dataset due to limitations observed with CIFAKE, which primarily contains fully synthetic images and does not reflect partial manipulations. This ensures better alignment between the dataset and our dual-task objective.

Paragraph 2 (Datasets considered but not used – with reasoning):
Several additional datasets were explored but not used due to compute constraints or methodological mismatch. For classification, GenImage was considered due to its diversity across multiple generative models, but its very large scale made it impractical under our compute constraints. CIFAKE was also used initially; however, it primarily contains fully synthetic images and failed to generalize to our task of detecting partial manipulations, leading us to move toward a custom dataset better aligned with our objective. For segmentation, the full TGIF2 dataset was highly relevant as it contains text-guided inpainting-based manipulations with localization annotations, but its large size and preprocessing requirements made it infeasible to use in full, so we adopted the SD2-FR subset instead. Additionally, datasets like CASIA 2.0 were not used because they focus on traditional manipulations such as copy-move and splicing, which do not reflect the artifacts introduced by modern diffusion-based generative models, resulting in a methodological mismatch with our problem setting.

Paragraph 3 (Evaluation Metrics & Framing):
We evaluate our system using both classification and segmentation metrics. For classification, we use Accuracy, accuracy on fake images and accuracy on real images to measure detection performance, while for segmentation we use Mean Intersection over Union (mIoU) to quantify localization quality and pixel-wise accuracy. Additionally, from an engineering standpoint, we measure inference latency to ensure the system can provide timely feedback. Importantly, our experimental setup is designed to evaluate how well a combined classification–segmentation pipeline performs in detecting partial manipulations without relying on frequency-domain priors, thereby directly testing the effectiveness of purely visual learning-based approaches.

Slide 4: Dataset and Evaluation Metrics

Datasets Used
CIFAKE: Used initially for baseline classification training.
SD2 FR: A subset of the TGIF2 dataset containing original SD2-generated full FAKE images. Primarily used to train our segmentation models (UNet and DeepLabV3Plus).
Custom Classification Dataset: A balanced dataset derived directly from the SD2 FR dataset explicitly to train and evaluate our classification models robustly.

Evaluation Metrics
Classification: Overall Accuracy, Fake Accuracy (Accuracy on Fake images), Real Accuracy (accuracy on real images).
Segmentation: mean Intersection over Union (mIoU), Pixel Accuracy.
Pipeline: Combined True Positive Rate (TPR) for the joint heuristic classification.

Slide 5: System Baselines and Initial Failure Case - Custom dataset

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

Slide 6: Results of updated pipeline

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

Slide 7: Error Rectification via Heuristics

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

Slide 8: Next Steps and Summary

Future Work and Direction
1. Joint Training: Instead of training and running classification and segmentation separately (which yields lower standalone classification accuracy), we plan to investigate end-to-end joint training to allow the networks to share spatial and contextual features.
2. Frequency and Noise Domain Features: Taking inspiration from State-Of-The-Art models like DIRE and FIRE, we intend to introduce frequency domain or noise-residual features into our models, rather than relying solely on RGB pixel data.
3. Enhanced Web Interface: For the final presentation, our live evaluation platform will be updated to allow users to toggle between current models, our upcoming joint-training models, and established SOTA baseline models for side-by-side performance comparisons.

Contributions:
1. Divyanshu: research on segmentation datasets, fine-tune and evaluate segmentation models, web-interface
2. Garvit: research on classification datasets, fine-tune and evaluate classification models, custom classification dataset, evaluate overall pipeline, web-interface
3. Rewant: Research relevant models and datasets
4. Tanish: Research relevant models and datasets

Future tasks:
1. Divyanshu: Research relevant datasets and implement joint training, try frequency and noise domain based improvements over baseline models
2. Garvit: Research relevant datasets and implement joint training, try frequency and noise domain based improvements over baseline models
3. Rewant: Add SOTA models on web interface, compare and analyse their results against baseline models
4. Tanish: Add SOTA models on web interface, compare and analyse their results against baseline models

Slide 9: Demo

Live demonstration of the AuthentiLens Streamlit web application.
Showcasing simultaneous multi-model inference, visualization of segmentation probability masks vs binary masks, and real-time system performance profiling.