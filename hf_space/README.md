---
title: AuthentiLens
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Detect AI-inpainted photos and localise the edited region
---

# AuthentiLens

**Detects AI-inpainted images and highlights the regions that were edited.**

Upload a photo. An EfficientNet-B0 classifier decides FAKE vs REAL, and a
DeepLabV3+ segmentation network predicts which pixels were inpainted. When the
classifier says REAL but the segmenter flags more than 30% of the pixels, the
pipeline overrides the call to FAKE — that override is what catches partial
edits the classifier misses.

This Space runs the **April pipeline**: EfficientNet-B0 (balanced patches,
lr 2.5e-5) + DeepLabV3+, pixel threshold 0.7, override 30%. It reached
**96.70% combined TPR** on the 1,029-image SD2-FR test set.

## Scope and limits

- Built for **diffusion inpainting on photoreal images** (Stable Diffusion 2 /
  SDXL edits of real photos). Not trained for GAN images, fully synthetic
  images or artwork.
- The pipeline was evaluated on a **fake-only** test set, so its
  **false-positive rate on real photos has not been measured**. Treat results
  as indicative, not as proof.
- The classifier alone is only 71.4% accurate on that set, and the task is
  genuinely hard — human majority vote scored 44.2% on the same images.

## How it is built

The Space does not contain a copy of the demo. `app.py` downloads the weights
from [iamgarvit/authentilens-weights](https://huggingface.co/iamgarvit/authentilens-weights)
and then runs `app/streamlit_app.py` from the
[AuthentiLens repository](https://github.com/iamgarvit/AuthentiLens) unchanged,
so the deployed demo and the local one are the same code.

Course project for **CSE344 Computer Vision, IIIT Delhi** (team Aperture):
Divyanshu Yadav, Garvit, Rewant Anand and Tanish Bachhas.
