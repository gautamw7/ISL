---
# 🧪 Experiments.md — ISL Recognition Project

## 1️⃣ Experiment Objectives

* Assess feasibility of **scalable Indian Sign Language recognition** using **landmark-based models**.
* Test:

  * Bounding box normalization.
  * Procrustes alignment.
  * DTW-based augmentation.
  * GRU-based temporal modeling.
* Evaluate **generalization to user-recorded videos** (domain shift).
* Test **fine-tuning viability** for small-class personalization.
* Document **negative findings** to highlight dataset limitations for future ISL work.

---

## 2️⃣ Dataset

* **Base Dataset:** ISL RGB videos with **45–262 labels**, \~110 frames per video.
* **Landmarks:**

  * 543 keypoints (x, y, z) per frame (flattened to 1629 features).
  * Extracted using **MediaPipe**.
* **Test Data:**

  * User-recorded short videos (2–3 sec) for "Beautiful", "Nice", etc.
  * Tested cross-domain generalization and fine-tuning.

---

## 3️⃣ Preprocessing Experiments

✅ **Padding:** All sequences to 110 frames.
✅ **Bounding Box Normalization:** Translate landmarks to top-left, scale by width, height, depth for invariance.
✅ **Procrustes Alignment:** Align each frame’s landmarks to a reference template to reduce domain mismatch.
✅ **DTW Augmentation:**

* Used average DTW distances to select top labels for augmentation.
* Increased dataset variability and temporal alignment robustness.

---

## 4️⃣ Model Experiments

### 🩻 Baseline:

* GRU-based bidirectional model on landmark sequences.
* Trained on 45 classes with categorical cross-entropy.

### 🛠️ Hyperparameter Search:

* Learning Rates: `1e-3`, `1e-4`, `1e-5`.
* Dropout: `(0, 0.1, 0.1)` and `(0, 0.1, 0.15)`.
* Patience: 10, 12 (early stopping).
* Batch size: 16.
* Epochs: 80 (full), 30–70 (fine-tuning).

### 🩻 Best Results:

```
Final Training Accuracy: 98.90%
Final Validation Accuracy: 92.31%
F1-score: 0.9208
Precision: 0.9411
Recall: 0.9222
AUC-ROC: 0.9985
```

---

## 5️⃣ Testing on User Videos

### Direct Testing (Without Fine-Tuning):

* Significant domain mismatch.
* Mean landmark distance:

  ```
  Mean distance: 7.14
  Min distance: 0.91
  Max distance: 30.56
  ```
* Visual scatter plots & heatmaps showed **pattern similarity but poor positional alignment**.

### Fine-Tuning (Without Freezing):

* **Accuracy on fine-tuning validation: 80%**.
* Post-fine-tuning performance on previous labels **dropped to 0%**, indicating **catastrophic forgetting**.

### Fine-Tuning (With Layer Freezing):

* Validation accuracy during fine-tuning reached **100%**.
* Overall model accuracy: **88.24%** post-fine-tuning.
* However, testing on original labels still resulted in **0% accuracy** for "Beautiful", "Nice", etc.
* Predicted the **same wrong label for all test samples**, indicating **overfitting on small fine-tuning data**.

---

## 6️⃣ Analysis & Insights

* Procrustes + bounding box normalization **reduced mean landmark distances** but did not fully solve domain mismatch.
* DTW augmentation improved temporal consistency.
* The GRU-based model performs well within dataset splits but **fails to generalize to user videos**.
* Fine-tuning on small personalized datasets without careful strategies leads to **catastrophic forgetting**.
* **Scalable ISL recognition requires a diverse, dedicated dataset**; small-scale landmark models are insufficient for real-world deployment.

---

## 7️⃣ Experiments Summary Table

| Experiment                       | Result                                       |
| -------------------------------- | -------------------------------------------- |
| Baseline GRU on padded landmarks | 92% val acc, high intra-dataset performance  |
| Bounding box normalization       | Improved consistency across samples          |
| Procrustes alignment             | Reduced positional domain gaps               |
| DTW-based augmentation           | Improved temporal alignment stability        |
| Fine-tuning (no freeze)          | Overfitting, catastrophic forgetting         |
| Fine-tuning (frozen layers)      | 100% val on few samples, 0% on prior classes |
| Testing on user-recorded videos  | Failed generalization                        |

---

## 8️⃣ Tools & Libraries Used

* **TensorFlow / Keras**: Modeling and training.
* **NumPy, Pandas**: Data manipulation.
* **scikit-learn**: Confusion matrices, F1, precision, recall.
* **Matplotlib, Seaborn**: Visualizations (ROC curves, scatter plots, heatmaps).
* **OpenCV / MoviePy**: Video frame extraction and processing.
* **MediaPipe**: Landmark extraction.

---

## 9️⃣ Next Steps

✅ **Prepare Paper: “Negative Findings on ISL Recognition with Landmark-Based Small Datasets.”**
✅ Draft methodology and discussion sections using these experiments.
✅ Archive plots and confusion matrices for the paper.
✅ Prepare “Things NOT to Do in ISL Recognition” conclusions with clear quantitative failures.
✅ Final testing of Procrustes + bounding box normalization with larger test batches if possible.

---

## 10️⃣ Metadata

* **Total messages (approx.):** 250+
* **Total words discussed (approx.):** 60,000+
* **Tone:** Debug-focused, iterative workflow alignment, negative findings exploration.
* **Date range:** until 2025-06-28

---

## ✅ Checklist of Potential Misses

Please confirm or upload if you want them integrated:

* [ ] Exact **hyperparameter tuning ranges** across all runs.
* [ ] **Scripts for landmark extraction** if needed in the appendix.
* [ ] **Detailed plots** for ROC, confusion matrix with label mappings.
* [ ] Notes on **hardware used (GPU/CPU RAM, etc.)**.
* [ ] Notes on **inference speed, model size** for deployment feasibility analysis.
* [ ] Any **formal reference list (BibTeX) entries** to track cited tools and methods.

---

