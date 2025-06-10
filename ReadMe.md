## Multi-Class Brain Tumor MRI Classification

**Group Ken9**  
**Members:**
- **Project Leader:** Balázs Ács  
- **Leader Code 1:** Rayan Atieh  
- **Leader Code 2:** Valters Eglitis  
- **Leader Report:** Rafal Dorsz  
- **Leader Presentation:** Ole Rotteveel  
- **Leader Administration:** Yann Klees  

---

## Introduction

This repository implements a **multi-class brain tumor classification** workflow on MRI data, specifically recognizing four classes: glioma, meningioma, pituitary tumor, and no tumor. It covers the entire pipeline from **data preprocessing** and **augmentation** to **training** multiple CNN models (custom CNNs and optional transfer learning). The project addresses research questions about preprocessing strategies, data augmentation, hyperparameter search (HPC), and model architectures (LpCNN, ResNet18, etc.).

---

## Table of Contents
1. [Environment & Installation](#environment--installation)
2. [Repository Structure](#repository-structure)
3. [Main Features](#main-features)
   - [Preprocessing](#a-preprocessing)
   - [Augmentation](#b-augmentation)
   - [Dashboard](#c-dashboard)
   - [Compare Tab](#d-compare)
   - [AutoSearch (HPC)](#e-autosearch-hpc)
   - [Info Tab](#f-info-tab)
   - [Inference](#g-inference)
4. [Typical Workflow](#typical-workflow)
5. [Notebooks](#notebooks)
6. [Running the GUI](#running-the-gui)
7. [Next Steps & Limitations](#next-steps--limitations)
8. [Citation & Acknowledgments](#citation--acknowledgments)

---

## Environment & Installation

**Recommended versions (Python 3.9+):**
- `PyTorch 1.13+`, `torchvision 0.14+`
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `sklearn`
- `pillow` (PIL)
- `nibabel` (for NIfTI if needed)
- `PyQt5` (GUI)
- `tqdm` (for progress bars in notebooks)


Install via:
```bash
pip install -r requirements.txt
```
## Main Features

### A. Preprocessing

Under **PreprocessingTab** in the GUI:

1. Select source (`data/archive`) and destination (e.g. `cleaned_data`) folders.
2. Optionally remove duplicates by hashing.
3. Apply histogram equalization (CLAHE), blur, sharpen.
4. Resize images (e.g. 128×128, 224×224).
5. Start the process. New images will be saved in `cleaned_data/...`.

### B. Augmentation

Under **AugmentationTab**:

* **Preview**: See a single image with random transforms (rotate, brightness, gamma, flip, etc.).
* **Bulk**: For each class, create copies up to a target count. E.g. if you set target 2000 and a class has only 500 images, the code randomly applies transformations until that class reaches 2000. Results go into `augmented_data/...`.

### C. Dashboard

A read-only **DashboardTab** to view dataset distributions:

* Bar charts of per-class counts.
* Histograms of pixel intensities.
* Box plots comparing subsets (Training vs. Testing).

### D. Compare

The **CompareTab** locates the same filename across **raw**, **cleaned**, and **augmented** sets and shows them side-by-side. Great for confirming how an image was transformed.

### E. AutoSearch (HPC)

**auto\_search\_tab.py** does a hyperparameter search:

* Chooses combos of model variant (binary, threeclass, simple), dropout, weight decay, early stopping, class weighting, etc.
* Trains each combo for “mini-epochs.”
* Saves best combos in `hpc_results.json`.
* You can train a final model at full epochs after HPC.

### F. Info Tab

Contains usage instructions, gamma slider demonstrations, histogram eq demos, and an overview of the entire pipeline.

### G. Inference

**inference\_tab.py**:

* Load a model checkpoint (from HPC combos or custom path).
* Single-Image Inference: predict tumor class. Optional Grad-CAM to visualize important regions.
* Batch Inference: process a folder of images, optionally exporting to CSV.


## Typical Workflow

1. **Preprocessing**

   * Copy raw → cleaned, remove duplicates, apply histogram eq/resizing, etc.
2. **Augmentation** (if needed)

   * Either preview or generate many random copies to handle class imbalance.
3. **Dashboard**

   * Check class distributions, intensities, resolution stats.
4. **Compare**

   * Confirm a single image’s transformations across raw/cleaned/aug.
5. **AutoSearch**

   * Launch HPC to find a best model variant, dropout, weight decay, etc.
   * Train a final model with those best hyperparams.
6. **Inference**

   * Use the final model to classify new images (single or batch).
   * Optionally visualize with Grad-CAM.

## Notebooks

If you prefer notebooks, see the `notebooks/` folder there we conducted all our experiments for the reserach questions and some dataset experiments.

## Running the GUI

1. Ensure dependencies installed:

   ```bash
   pip install -r requirements.txt
   ```
2. Place raw MRI data under `data/archive/Training` / `Testing`.
3. Run the main GUI:

   ```bash
   python -m main
   ```

   or

   ```bash
   python main.py
   ```
## Citation & Acknowledgments

* **Dataset**: [Kaggle Brain Tumor MRI Data](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
* **LpConv** idea: Adapted from friend’s code / references in `lpconv.py`.
* **ResNet** from PyTorch’s [torchvision.models](https://pytorch.org/vision/stable/models.html).