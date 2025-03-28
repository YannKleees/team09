# README

## Multi-Class Brain Tumor MRI Classification: A Machine Learning Approach

**Group Ken9**  
**Group Members:**  
- **Project Leader:** Balázs Ács  
- **Leader Code 1:** Rayan Atieh  
- **Leader Code 2:** Valters Eglitis  
- **Leader Report:** Rafal Dorsz  
- **Leader Presentation:** Ole Rotteveel  
- **Leader Administration:** Yann Klees  

### Introduction
This project prototype addresses **multi-class brain tumor classification** using MRI data. It aligns with the team’s broader research goals:

“How can machine learning be leveraged to predict and classify brain tumor types from MRI scans, effectively addressing challenges such as image variability and tumor heterogeneity?”*

Concretely, we handle **four classes**: glioma, meningioma, pituitary tumor, and no tumor. Our pipeline manages the entire **data lifecycle** from raw images to preprocessing, augmentation, and training.

### Methods & Pipeline
1. **Data Preprocessing**  
   - **Copying & Cleaning**: We copy the original dataset into a dedicated `cleaned_data/` folder so the raw data remains untouched.  
   - **Optional Duplicate Removal**: Identifies and removes duplicates based on file hash checks to ensure unique samples.  
   - **Resizing**: User-selectable (e.g., `128×128`, `160×160`, `224×224`) for standardizing input size.  
   - **Normalization**: Either **z-score** (with optional re-scaling to [0..1]) or **min-max** (0..1) to reduce scanner-induced artifacts.

2. **Data Augmentation**  
   - **Preview Mode**: Quickly test a single image with random rotations, brightness/contrast jitter, gamma corrections, and optional min-max normalization.  
   - **Bulk Augmentation**: Generate multiple new images, 2–20 copies per image for the entire dataset in `cleaned_data/`, saving results in `augmented_data/`.  
   - Helps address **RQ2**: “How does data augmentation improve robustness?”

3. **Dashboard & Comparison**  
   - **Dashboard**: Analyze folder distribution, sample intensities, and plot histograms. Compare different normalization approaches to see how intensities shift.  
   - **Comparison**: Browse the same image filename across raw vs. cleaned vs. augmented sets side-by-side.

4. **Training**  
   - **Custom CNN**: A small **PyTorch**-based CNN for demonstration.  
   - **User Controls**: Epochs, batch size, and learning rate. The training runs in a **QThread** so the GUI remains responsive.  
   - **Real-time Logs & Progress**: See logs and a progress bar as training proceeds (addresses **RQ3** about hyperparameter setups).

### Relevance to the Project Plan
This prototype addresses the **core steps** outlined in our proposal:

- **RQ1** & **RQ2**: Implement standard preprocessing (resizing, normalization) and controlled augmentation.  
- **Hyperparameter Tuning (RQ3)**: We’ve built a modular CNN training loop that can be extended with hyperparameter searches.  
- **Performance Monitoring (RQ6)**: We log per-epoch validation accuracy and can add additional metrics (precision, recall, F1).  
- **Comparison with Transfer Learning (RQ5)**: The code includes a placeholder for a ResNet-based approach, forming the basis for future expansions.

### Next Steps

1. **Implement Transfer Learning (ResNet or other)**
   - Compare a pre-trained ResNet-based pipeline with the custom CNN to fulfill **RQ5**.
2. **Advanced Metrics**
   - Add confusion matrices, per-class F1-scores, and a “Plot Metrics” button.
3. **Hyperparameter Search**
   - Automate tuning for learning rate, batch size, and data augmentation parameters.
4. **Inference GUI**
   - Possibly add an “Upload & Predict” tab for a final user-facing classification interface.

### Typical Workflow

1. **Preprocessing Tab**  
   - Copy and optionally remove duplicates  
   - Choose resize & normalization  
   - Click “Preprocess.”

2. **Augmentation Tab**  
   - Preview augmentation or generate multiple versions  
   - Click “Apply Bulk Augmentation.”

3. **Dashboard Tab**  
   - Inspect distribution & intensity stats  
   - Compare normalization methods

4. **Comparison Tab**  
   - Browse side-by-side images across data folders

5. **Training Tab (placeholder)**  
   - Adjust epochs, batch size, learning rate  
   - Click “Train Custom CNN.”

### How to Run
1. **Install dependencies**:
   pip install -r requirements.txt

**1. Prepare Data**  
Place your raw MRI data under `data/archive` and `data/archive` (or adjust paths in the code). if not already there

**2. Launch the Application**  
python main.py