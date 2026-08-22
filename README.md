<div align="center">

# 👗 Fashion MNIST Image Classifier

### HOG Feature Extraction + Linear SVM · Streamlit Web UI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[![Stars](https://img.shields.io/github/stars/SoumiryaSarangi/Image-Classifier-for-fashion-mnist-hog-svm?style=social)](https://github.com/SoumiryaSarangi/Image-Classifier-for-fashion-mnist-hog-svm)
[![Forks](https://img.shields.io/github/forks/SoumiryaSarangi/Image-Classifier-for-fashion-mnist-hog-svm?style=social)](https://github.com/SoumiryaSarangi/Image-Classifier-for-fashion-mnist-hog-svm)
[![Issues](https://img.shields.io/github/issues/SoumiryaSarangi/Image-Classifier-for-fashion-mnist-hog-svm?style=social)](https://github.com/SoumiryaSarangi/Image-Classifier-for-fashion-mnist-hog-svm)

<br/>

A high-performance image classification pipeline that classifies **Fashion MNIST** images using **Histogram of Oriented Gradients (HOG)** feature extraction and a **Linear Support Vector Machine (SVM)** — built with a clean, modular Python architecture and an interactive **Streamlit web UI**.

[Getting Started](#-installation) · [Usage Guide](#-usage-guide) · [Evaluation](#-evaluation-metrics) · [Contributing](#-contributing)

<br/>

<img src="https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/fashion-mnist-sprite.png" alt="Fashion MNIST Sprite" width="600"/>

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Requirements](#-requirements)
- [How to Run](#-how-to-run)
- [Usage Guide](#-usage-guide)
- [Streamlit Web UI](#-streamlit-web-ui)
- [Evaluation Metrics](#-evaluation-metrics)
- [What Was Added](#-what-was-added)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Author](#-author)

---

## 🔍 Project Overview

Fashion MNIST is a dataset of **Zalando's article images** consisting of **70,000 grayscale images** (28×28 pixels) across **10 fashion categories**. This project implements a complete machine learning pipeline to classify these images using classical computer vision techniques.

### 🎯 How It Works

```
Input Image (28×28) → Preprocessing → HOG Feature Extraction → Linear SVM → Predicted Class
```

| Step | Description |
|------|-------------|
| **1. Load Data** | Fetches Fashion MNIST via `tensorflow.keras.datasets` |
| **2. Preprocess** | Normalizes pixel values to `[0, 1]` and reshapes with channel dimension |
| **3. Extract HOG** | Computes Histogram of Oriented Gradients (4×4 px/cell, 2×2 cells/block) |
| **4. Train SVM** | Fits a Linear SVM on the extracted HOG feature vectors |
| **5. Evaluate** | Generates accuracy, confusion matrix, classification report & confusion analysis |
| **6. Predict** | Classifies custom images via the CLI or Streamlit web UI |
| **7. Tune** | Optional hyperparameter tuning via GridSearchCV on a stratified subsample |

### 🏷️ Supported Classes

| Label | Class | Label | Class |
|:-----:|-------|:-----:|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

---

## ✨ Features

<table>
<tr>
<td>

### 🏗️ Architecture
- Modular Python package structure
- Separate modules for each pipeline stage
- Clean separation of concerns

### 🧠 Machine Learning
- HOG feature extraction (`skimage`)
- Linear SVM classifier (`scikit-learn`)
- Train & test accuracy reporting
- Hyperparameter tuning via `GridSearchCV`
- Confusion matrix analysis (top misclassified pairs)

### 💾 Persistence
- Model saving/loading via `joblib`
- HOG feature caching to avoid recomputation
- Automatic detection of saved artifacts

</td>
<td>

### 📊 Evaluation
- Confusion Matrix visualization
- Classification Report generation
- Precision, Recall & F1-score per class
- Auto-saving of all evaluation outputs

### 🖼️ Prediction
- Custom image prediction via CLI & Streamlit
- OpenCV-based image preprocessing
- Prediction visualization grid

### 🖥️ Interface
- Interactive continuous command-line menu
- Premium Streamlit web UI (glassmorphism design)
- Clean console output formatting
- Guided user prompts

</td>
</tr>
</table>

---

## 🛠️ Technologies Used

| Category | Technology | Version |
|----------|-----------|---------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) Python | `3.10+` |
| **ML Framework** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) scikit-learn | `1.9.0` |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white) OpenCV | `5.0.0` |
| **Feature Extraction** | ![scikit-image](https://img.shields.io/badge/scikit--image-00A3E0?style=flat-square) scikit-image | `0.26.0` |
| **Data Handling** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) NumPy | `2.5.1` |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square) Matplotlib | `3.11.1` |
| **Dataset Source** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) TensorFlow (Keras) | `2.21.0` |
| **Serialization** | ![Joblib](https://img.shields.io/badge/Joblib-2C2D72?style=flat-square) Joblib | `1.5.3` |
| **Web UI** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) Streamlit | `1.45.0` |

---

## 📁 Folder Structure

```
Fashion-MNIST-HOG-SVM/
│
├── 📄 main.py                        # CLI entry point — orchestrates the full pipeline
├── 📄 app.py                         # Streamlit web UI for image prediction
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # Project documentation (you are here!)
│
├── 📂 src/                            # Source package
│   ├── __init__.py                    # Package initializer
│   ├── config.py                      # Centralized file paths configuration
│   ├── dataset.py                     # Fashion MNIST data loading
│   ├── preprocessing.py               # Normalization & reshaping
│   ├── features.py                    # HOG feature extraction
│   ├── model.py                       # SVM training, evaluation, tuning & persistence
│   ├── predict.py                     # Custom image prediction pipeline
│   ├── visualize.py                   # Sample & prediction visualizations
│   └── cache.py                       # HOG feature caching utilities
│
├── 📂 models/                         # Serialized artifacts
│   ├── image_classifier.joblib        # Trained SVM model
│   ├── train_hog.joblib               # Cached training HOG features
│   └── test_hog.joblib                # Cached testing HOG features
│
├── 📂 outputs/                        # Evaluation outputs
│   ├── confusion_matrix.png           # Confusion matrix heatmap
│   ├── classification_report.txt      # Per-class metrics + confusion analysis
│   └── predictions.png               # Prediction visualization grid
│
└── 📂 images/                         # Custom images for prediction
    ├── bag.png
    └── shoes.png
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10** or higher
- **pip** package manager
- **Git** (optional, for cloning)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/SoumiryaSarangi/Image-Classifier-for-fashion-mnist-hog-svm.git
cd Image-Classifier-for-fashion-mnist-hog-svm
```

**2. Create a virtual environment** _(recommended)_

```bash
python -m venv .venv
```

**3. Activate the virtual environment**

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The core dependencies for this project:

```
tensorflow>=2.21.0
scikit-learn>=1.9.0
scikit-image>=0.26.0
opencv-python>=5.0.0
numpy>=2.5.1
matplotlib>=3.11.1
joblib>=1.5.3
pillow>=12.3.0
streamlit>=1.45.0
```

> [!NOTE]
> TensorFlow is used **only** for loading the Fashion MNIST dataset via `keras.datasets`. The classification itself is fully powered by scikit-learn.

---

## 🚀 How to Run

### CLI Mode

```bash
python main.py
```

On first run, the pipeline will:

1. 📥 Download the Fashion MNIST dataset (automatic via Keras)
2. 🔄 Preprocess images (normalize + reshape)
3. 📐 Extract HOG features (and cache them for future runs)
4. 🧠 Train the Linear SVM classifier (and save the model)
5. 📋 Present the interactive menu

On subsequent runs, cached HOG features and the saved model are **loaded automatically**, making startup significantly faster.

### Streamlit Web UI

```bash
pip install streamlit
streamlit run app.py
```

The web UI loads the saved model (trained via the CLI) and provides a drag-and-drop image prediction interface. See [Streamlit Web UI](#-streamlit-web-ui) for details.

---

## 📖 Usage Guide

After launching the program, you'll see the interactive menu:

```
Choose an option:
1. Evaluate Model
2. Predict Custom Image
3. Tune Hyperparameters
4. Exit
```

### Option 1 — Evaluate Model

Runs the trained model against both training and testing sets:

- Prints **training accuracy** and **testing accuracy**
- Displays the **confusion matrix** as a heatmap
- Generates and saves the **classification report**
- **Analyzes the confusion matrix** — identifies and prints the **top-3 most confused class pairs** (e.g. _"Shirt confused as T-shirt/top: 101 times"_) and appends the analysis to `outputs/classification_report.txt`
- Shows a **prediction visualization** grid (true vs. predicted labels)

All evaluation outputs are automatically saved to the `outputs/` directory.

### Option 2 — Predict Custom Image

Provide the path to any fashion image:

```
Enter image path: images/shoes.png
```

The pipeline will:
1. Load and convert the image to grayscale
2. Resize to 28×28 pixels
3. Normalize and extract HOG features
4. Predict the fashion category
5. Display the image with the prediction via matplotlib

> [!TIP]
> For best results, use images with a **clean background** and a **single centered item**. The model was trained on 28×28 grayscale images, so high-contrast images work best.

### Option 3 — Tune Hyperparameters

Runs `GridSearchCV` to find optimal SVM hyperparameters:

- Takes a **stratified 8,000-sample subset** of the training data (configurable) to keep grid search fast and RAM-friendly
- Searches over `C = [0.01, 0.1, 1, 10]` × `kernel = ["linear", "rbf"]` with **3-fold cross-validation**
- Prints the **best params** and **best CV accuracy**
- **Retrains** a fresh SVM with the best params on the **full 60,000-sample training set**
- Prompts whether to **save** the tuned model (overwrites the existing saved model) or discard it

> [!NOTE]
> Tuning is intentionally **opt-in** and only runs when you pick this menu option — it is never triggered automatically on startup.

### Option 4 — Exit

Cleanly exits the program.

---

## 🌐 Streamlit Web UI

The project includes a **premium Streamlit web app** (`app.py`) for browser-based prediction — no terminal needed.

### Running the Web UI

```bash
pip install streamlit
streamlit run app.py
```

> [!IMPORTANT]
> The web UI requires a **pre-trained model** (`models/image_classifier.joblib`). Run `python main.py` first to train and save the model before launching the Streamlit app.

### Web UI Features

| Feature | Description |
|---------|-------------|
| **Model loading** | Loads the saved SVM model once via `@st.cache_resource` (no reloading on each interaction) |
| **Drag & drop upload** | Accepts PNG, JPG, JPEG images via a styled file uploader |
| **Live prediction** | Runs the same HOG pipeline as the CLI and displays the predicted class |
| **Two-column layout** | Uploaded image on the left, prediction result card on the right |
| **Confidence display** | Shows class probabilities if the model supports `predict_proba` (graceful fallback if not) |
| **Premium design** | Glassmorphism cards, gradient backgrounds, Inter font, animated hero, custom CSS |

### How It Works

`app.py` imports `load_model()`, `model_exists()`, `preprocess_data()`, and `extract_hog_features()` directly from the existing `src/` package — no model-loading or preprocessing logic is duplicated. The only duplicated element is the `CLASS_NAMES` list (since importing `main.py` would trigger the full pipeline).

---

## 📊 Evaluation Metrics

The model is evaluated using industry-standard classification metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall ratio of correct predictions |
| **Precision** | How many selected items are relevant (per class) |
| **Recall** | How many relevant items are selected (per class) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Visual breakdown of predictions vs. true labels |

### Sample Classification Report

```
              precision    recall  f1-score   support

           0     0.8425    0.8560    0.8492      1000
           1     0.9890    0.9710    0.9799      1000
           2     0.8119    0.8360    0.8238      1000
           3     0.8871    0.8950    0.8910      1000
           4     0.8089    0.8370    0.8227      1000
           5     0.9721    0.9570    0.9645      1000
           6     0.7214    0.6830    0.7017      1000
           7     0.9393    0.9620    0.9505      1000
           8     0.9754    0.9720    0.9737      1000
           9     0.9542    0.9540    0.9541      1000

    accuracy                         0.8923     10000
   macro avg     0.8902    0.8923    0.8911     10000
weighted avg     0.8902    0.8923    0.8911     10000
```

> [!NOTE]
> Results may vary slightly depending on system and library versions. The metrics above are representative of a typical run.

---

## 🔨 What Was Added

### 🧱 Base Project (Starting Point)

The original implementation provided the following core functionality:

- Load the Fashion MNIST dataset
- Perform basic image preprocessing
- Extract HOG (Histogram of Oriented Gradients) features
- Train a Linear SVM classifier
- Evaluate the model using training and testing accuracy
- Display a few sample Fashion MNIST images

### 🔧 Improvements to Existing Functionality

- Refactored the original pipeline into reusable, single-responsibility modules
- Added model + HOG feature persistence via Joblib — retraining and re-extraction only happen once instead of on every run
- Expanded evaluation beyond accuracy with a full classification report and confusion matrix visualization
- Automatically saved all evaluation outputs to the `outputs/` directory
- Improved console output with clearer progress messages and organized reporting
- Centralized all file paths into a single `src/config.py` module for better maintainability
- Updated the interactive CLI to run in a continuous loop, returning to the main menu after each action
- Replaced OpenCV's `cv2.imshow` with `matplotlib.pyplot` in the prediction module to fix RGB color rendering issues and improve reliability
- Deduplicated model loading logic to rely on a single source of truth in `src/model.py`
- Added **confusion matrix analysis** — `analyze_confusion_matrix()` automatically identifies the top-3 most confused class pairs after evaluation and appends the analysis to the classification report
- Added **invalid input handling** — non-menu inputs now print a message and re-prompt instead of silently exiting

### ✨ New Features

- Added prediction support for external custom images via an interactive command-line prompt
- Added prediction visualization displaying true vs. predicted labels side by side
- Added automatic export of prediction screenshots, the classification report (text), and the confusion matrix (image)
- Added **hyperparameter tuning** via `GridSearchCV` — searches over `C` and `kernel` on a stratified subsample, then retrains on the full training set with the best params. Exposed as CLI menu option 3
- Added a **Streamlit web UI** (`app.py`) — a premium, glassmorphism-styled browser interface for drag-and-drop image prediction, reusing the existing `src/` pipeline without duplicating logic

### 🏗️ Design & Architecture

- Reorganized the project into a modular `src/` package
- Separated responsibilities into dedicated modules: `dataset`, `preprocessing`, `features`, `model`, `visualize`, `predict`, `cache`, and `config`
- Added dedicated `models/`, `outputs/`, and `images/` directories
- Added a centralized `requirements.txt` for dependency management
- Added a professional `.gitignore` to exclude generated files and virtual environments
- Added comprehensive project documentation in `README.md`
- Organized generated artifacts separately from source code for a cleaner project structure

---

## 📸 Screenshots

<details>
<summary><b>🖼️ Click to expand screenshots</b></summary>

<br/>

### Confusion Matrix

> The confusion matrix is saved to `outputs/confusion_matrix.png` after evaluation.

![Confusion Matrix](outputs/confusion_matrix.png)

### Prediction Visualization

> True labels vs. predicted labels for sample test images, saved to `outputs/predictions.png`.

![Predictions](outputs/predictions.png)

### Classification Report

The detailed classification report can be found here.

![View Classification Report](outputs/classification_report.txt)
</details>

> [!TIP]
> Run **Option 1 (Evaluate Model)** to generate all screenshot outputs automatically in the `outputs/` directory.

---

## 🔮 Future Improvements

- [x] 🌐 ~~Add a web-based UI using Streamlit or Gradio~~ — ✅ Streamlit UI added (`app.py`)
- [x] 🔧 ~~Hyperparameter tuning with GridSearchCV / RandomizedSearchCV~~ — ✅ `tune_model()` with GridSearchCV added
- [x] 📈 ~~Experiment with RBF / Polynomial SVM kernels~~ — ✅ Grid search includes both `linear` and `rbf` kernels
- [ ] 🧪 Cross-validation for more robust evaluation
- [ ] 🧹 Data augmentation (rotation, flip, noise)
- [ ] 📦 Docker containerization for reproducibility
- [ ] 🤖 Compare with CNN-based approaches (TensorFlow / PyTorch)
- [ ] 📊 Add ROC-AUC curves and per-class visualizations
- [ ] ⚡ GPU-accelerated HOG extraction
- [ ] 🧬 Experiment with PCA dimensionality reduction before SVM

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Soumirya Sarangi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👤 Author

<table>
<tr>
<td align="center">

**Soumirya Sarangi**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SoumiryaSarangi)


</td>
</tr>
</table>

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

Made with ❤️ and Python

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

</div>
