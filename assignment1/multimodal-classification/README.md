# Multimodal Classification with CrisisMMD

## 1. Problem Definition

- Use the **CrisisMMD dataset** to perform **Humanitarian classification** based on both:
  - Image
  - Tweet text

- Approach:
  - Use a **Vision-Language Model (VLM)** such as:
    - Qwen-VL 2B (preferred)
    - Qwen-VL 7B (fallback if 2B is unavailable)

- Perform classification under two settings:
  - **One-shot classification**
  - **Few-shot classification**

- Evaluate model under different quantization levels:
  - **4-bit**
  - **8-bit**

---

## 2. Objectives

- Compare:
  - One-shot vs Few-shot performance
  - 4-bit vs 8-bit quantization

- Evaluate:
  - Performance metrics
  - Inference time
  - Resource efficiency

- Use evaluation metrics:
  - Confusion Matrix
  - F1 Score
  - AUC

---

## 3. Implementation Steps

### 3.1 Data Preparation

- Due to class imbalance, merge categories as follows:

> Specifically:
> - Merge `injured or dead people` + `missing or found people` → `affected individuals`
> - Merge `vehicle damage` → `infrastructure and utility damage`
>
> → Final: **5 classes for humanitarian classification**

---

### 3.2 Exploratory Data Analysis (EDA)

Perform EDA to understand dataset characteristics:

- Label distribution
- Tweet length distribution
- Compare tweet lengths across classes
- Word cloud visualization
- Most frequent words per class

---

### 3.3 Model Preparation

- Implement a model wrapper class with the following features:

#### Requirements:
- Input:
  - Prompt
  - Image
  - Text

- Output:
  - Predicted class

- Configurable:
  - Quantization level (4-bit / 8-bit)

---

### 3.4 Classification Pipeline

- Design a classification flow that includes:

#### Functionalities:
- Measure **inference time per sample**
- Store results in a structured format (e.g., CSV)

#### Output file should include:
- `y_true`
- `y_pred`
- `image_path`
- `tweet_text`
- `inference_time`
- Any additional metadata

---

### 3.5 Evaluation

#### Metrics:
- Confusion Matrix
- F1 Score
- AUC

#### Comparisons:
- One-shot vs Few-shot
- 4-bit vs 8-bit quantization (for one-shot setting)

#### Error Analysis:
- Analyze misclassified samples
- Visualize selected failure cases

---

## 4. Code Requirements

- All code must:
  - Be written in **English**
  - Include:
    - Comments
    - Docstrings

- Follow best practices:
  - Clean, readable, maintainable code
  - Follow **SOLID principles**

- Visualization:
  - Prefer using **seaborn**

- Model loading:
  - Prefer models from **Hugging Face**

