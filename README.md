# Skin Lesion Boundary Detector

A coursework prototype for the AkiraChix DAS Medical Imaging ML Project. This tool takes a dermoscopic photo of a skin lesion and predicts a pixel-level mask outlining the lesion's boundary, using a U-Net convolutional neural network trained on the ISIC 2016 Challenge dataset.

> **This is a student coursework project, not a medical device. It has not been clinically validated and must never be used to make or support a real medical decision. It is not safe to use on images outside the type it was trained on (see Limitations below).**

-

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Problem Type](#problem-type)
3. [Dataset](#dataset)
4. [Model](#model)
5. [Results](#results)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Running the App](#running-the-app)
9. [How to Use the App](#how-to-use-the-app)
10. [Preprocessing Pipeline (Important)](#preprocessing-pipeline-important)
11. [Limitations](#limitations)
12. [Ethical Considerations](#ethical-considerations)
13. [Troubleshooting](#troubleshooting)

-

## What This Project Does

Given a single dermoscopic photo of a skin lesion, the app:
1. Accepts an uploaded image (JPG or PNG)
2. Resizes and normalizes it to match the model's training data
3. Runs it through a trained U-Net segmentation model
4. Overlays the model's predicted lesion boundary on the image in red
5. Reports what percentage of the image the model believes is lesion, in plain English

The goal is to demonstrate a working, end-to-end medical image segmentation pipeline  from raw data to a usable interface — not to produce a clinically reliable diagnostic tool.

## Problem Type

- **Body part / region:** Skin
- **Imaging modality:** Dermoscopic photo
- **Problem type:** Segmentation (pixel-level boundary detection, not classification of what the lesion is)
- **Target structure:** The visible boundary of a skin lesion within the image

## Dataset

- **Source:** [ISIC 2016 Challenge, Task 1  Lesion Segmentation](https://challenge.isic-archive.com/data/#2016)
- **License:** CC-0 (public domain)
- **Training data:** 900 dermoscopic JPEG images with matching binary PNG ground-truth masks (expert-annotated lesion boundaries)
- **Test data:** 379 dermoscopic JPEG images with matching binary PNG ground-truth masks, held out and never used during training
- **Validation split:** 15% of the training set (135 images) was set aside for validation during training; the remaining 765 images were used for actual training
- **Note on leakage:** The train/test boundary was set by ISIC's own official split. No image or lesion is known to appear in more than one of the training, validation, or test sets.

## Model

- **Architecture:** A custom, compact U-Net (encoder-decoder convolutional network with skip connections), built specifically for this project rather than using a pretrained backbone
- **Input:** 128×128×3 RGB image, normalized to a 0.0–1.0 pixel range
- **Output:** 128×128×1 array of values between 0 and 1, where each value represents the model's confidence that a pixel belongs to the lesion
- **Total parameters:** 487,297
- **Loss function:** Binary cross-entropy
- **Optimizer:** Adam
- **Training:** 25 epochs, batch size 16, trained on CPU

## Results

Evaluated on the 379-image held-out test set:

| Metric | Score |
|-|-|
| Mean Dice score | 0.884 |
| Mean IoU score | 0.810 |

- **Best individual prediction:** Dice = 0.982  the model traced the lesion boundary almost exactly.
- **Worst individual prediction:** Dice = 0.061  inspection showed the true ground-truth mask for this image was very small, and Dice score is known to become unstable (large apparent drop) on very small masks even when the visual prediction is reasonably close. This suggests part of the "worst case" is a metric sensitivity issue rather than a complete model failure.
- **Observed real-world quirk:** In manual testing, the model has occasionally produced a small false-positive patch in areas with dense hair coverage  likely mistaking overlapping dark hair strands for lesion-like boundary features.

## Project Structure

```
computer-vision-work/
├── images.ipynb                              # Full training/evaluation notebook (Part A)
├── app.py                                     # Streamlit diagnostic portal (Part B)
├── requirements.txt                           # Exact package versions used
├── README.md                                  # This file
├── unet_skin_lesion_model.h5                  # Saved trained model
├── ISBI2016_ISIC_Part1_Training_Data/         # 900 training images
├── ISBI2016_ISIC_Part1_Training_GroundTruth/  # 900 training masks
├── ISBI2016_ISIC_Part1_Test_Data/             # 379 test images
└── ISBI2016_ISIC_Part1_Test_GroundTruth/      # 379 test masks
```

## Installation

These steps assume a Linux machine with `conda` (Miniconda/Anaconda) available. Python 3.10 was used for development.

1. **Create an isolated environment** (recommended, to avoid permission or version conflicts with other Python setups on the machine):
   ```bash
   conda create -n skinapp python=3.10 -y
   conda activate skinapp
   ```

2. **Install dependencies** from the provided requirements file:
   ```bash
   python -m pip install -r requirements.txt
   ```
   Using `python -m pip` (rather than a bare `pip` command) ensures the packages install into the currently active `skinapp` environment, and avoids permission errors caused by a stray system-wide `pip` on some machines.

3. **Confirm the saved model file** (`unet_skin_lesion_model.h5`) is present in the same folder as `app.py`. If you're setting this up from scratch without the pre-trained model, run through `images.ipynb` first to train and export it.

## Running the App

With the `skinapp` environment activated and in the project folder:

```bash
python -m streamlit run app.py
```

This should automatically open the app in your default browser at `http://localhost:8501`. If it doesn't open automatically, copy that URL into your browser manually.

To stop the app, go back to the terminal and press `Ctrl+C`.

## How to Use the App

1. Click **Upload** and select a dermoscopic image in JPG or PNG format.
2. The app will display your uploaded image.
3. Below it, the app will display the same image with the model's predicted lesion boundary overlaid in semi-transparent red.
4. A message underneath states what percentage of the image the model believes is lesion.
5. If the model doesn't detect a clear lesion (very low predicted coverage), the app will say so explicitly rather than forcing a confident-looking but meaningless answer.

For best results, use dermoscopic-style images similar to the ISIC dataset closely cropped, dermoscope-captured photos of a single lesion. Regular phone camera photos or images of unrelated body parts will likely produce unreliable results (see Limitations).

## Preprocessing Pipeline (Important)

This exact pipeline is used both during model training and inside the app, and must never be changed independently in one place without updating the other:

1. Load the image and convert it to RGB.
2. Resize to 128×128 pixels.
3. Normalize pixel values by dividing by 255.0 (scaling the 0–255 range to 0.0–1.0).
4. No cropping, color adjustment, or augmentation is applied at inference time.
5. The model outputs a 128×128×1 array of values between 0 and 1.
6. A pixel is classified as "lesion" if its predicted value is greater than 0.5.

## Limitations

- **Training data scope:** The model was trained only on the ISIC 2016 dataset dermoscope-captured images from a specific set of clinical sources. Performance on images captured with different equipment, lighting, skin tones, or populations not represented in this dataset is untested and likely worse.
- **Small dataset size:** With only 900 training images, the model has limited exposure to rare or unusual lesion presentations.
- **Hair and obstruction sensitivity:** The model can occasionally misidentify dense hair coverage as lesion-like boundary features, producing small false-positive regions.
- **Not validated for clinical use:** This model has no mechanism for detecting malignancy or clinical significance it only outlines a boundary shape it has learned from pixel patterns.
- **Small-mask metric instability:** Dice score, the primary evaluation metric used, becomes less reliable and can swing sharply for images with very small true lesion areas.

## Ethical Considerations

- **False negatives** (the model missing part of a real lesion boundary) could cause a clinician to underestimate the true extent of a lesion.
- **False positives** (the model marking healthy skin as lesion) could cause unnecessary concern or follow-up.
- Neither type of error is acceptable as the basis for an actual diagnosis. This tool's output should be understood only as a "suggested outline" for illustrative/coursework purposes, never a clinical finding.
- Any real-world use of a tool like this would require validation on a much larger and more diverse dataset (multiple clinics, skin tones, and imaging equipment), formal clinical trials, regulatory approval, and integration into a clinician-led review process — not standalone, unsupervised use.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'streamlit'` (or tensorflow/pillow/numpy):** Your terminal is likely not using the `skinapp` environment's Python. Run `conda activate skinapp` first, and always install/run with `python -m pip install ...` / `python -m streamlit run app.py` rather than a bare `pip`/`streamlit` command, in case a different, stray installation exists elsewhere on your PATH.
- **`Permission denied` during installation:** This usually means pip is trying to write into a shared, locked system folder rather than your own environment. Creating and activating a dedicated `conda` environment (as shown above) resolves this, since it gives you a folder you fully own.
- **App loads but shows an error after uploading an image:** Confirm `unet_skin_lesion_model.h5` is in the same folder you're running `streamlit run app.py` from the app looks for it as a relative path.
- **Prediction looks blank or clearly wrong:** Confirm the uploaded image is a reasonably close-up dermoscopic-style photo similar to the training data; the model was not trained on general photography and will not generalize reliably to it.
