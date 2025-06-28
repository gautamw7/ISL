# 🧪 Experiments.md — ISL Recognition Project

---

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
## 11️⃣ Dataset Preparation

* **Datasets**:

  * Combined `holistic_labels_videos.pkl` and `rest_DTW_holistic_labels_videos.pkl`.
  * Landmarks extracted using **MediaPipe Holistic**, resulting in:

    * Shape per frame: `1629` features (`543` keypoints x `3` (x, y, z)).
    * Video sequence shapes: `(frames, 1629)`.
    * Padding/trimming standardized to `110` frames per video using interpolation if short and frame-skipping if long.
* **Statistics**:

  * Average frames per label calculated and plotted.
  * Typical frames per video: `110`.
  * Sequence length distribution and per-label frame distribution visualized.
* **Landmark Normalization**:

  * Explored:

    * **Chest-centered translation normalization**.
    * **Bounding box normalization** (scaling to body box width, height, depth).
    * **Procrustes alignment** (rotation + scaling + translation invariant alignment).
  * Final chosen pipeline: Bounding box normalization + Procrustes alignment for testing.
* **Labels**:

  * Top `30`, `45` label sets used for different scalability experiments.
  * Testing also performed on manually recorded **real-world test dataset** for "Beautiful" and "Nice".
* **GPU**:

  * **Kaggle T4 x 2** GPU environment used.
  * Average training time per run (8 models with 45 labels): \~15-20 minutes per configuration.

---

## 12️⃣ Hyperparameter Tuning

Explored hyperparameters:

* **Learning Rate**:

  * `0.0001, 0.00005` fixed for stability.
* **Dropout configurations**:

  * `(0, 0.1, 0.1)`
  * `(0, 0.1, 0.15)`
* **Patience for early stopping**:

  * `10`, `12`
* **Batch size**:

  * `16`
* **Epochs**:

  * Up to `80` (typically converged by `30-60`).

---

## 13️⃣ Model Architectures

* **Primary Model**:

  * GRU and Bidirectional GRU, LSTM, Bi-LSTM compared.
  * Typical pipeline:

    * Input: `(110, 1629)`
    * 2-3 layers GRU/LSTM/Bi-GRU with dropout.
    * Dense layers: `128 -> 64 -> 32`.
    * Output: Softmax over `N` labels (`N = 30, 45`).
* **Fusion Model (Planned, Not Run)**:

  * Hand landmarks: `21 keypoints x 3`, LSTM layers \[128, 64, 32].
  * Face landmarks: `68 keypoints x 3`, LSTM layers \[32, 16, 8].
  * Shoulder landmarks: `4 keypoints x 3`, LSTM layers \[8, 4, 2].
  * Concatenated and passed through Dense layers \[128, 64, 32].

---

## 14️⃣ Experiments & Results

### A) Scalability Experiments

* **30 Labels**:

  * Achieved validation accuracy: `~88-90%`.
  * AUC-ROC: `0.996+`
* **45 Labels**:

  * Achieved validation accuracy: `~92%` (without Procrustes).
  * After Procrustes alignment, slightly lower `85-88%`.
* **Test Data Real-World Generalization**:

  * Pre-finetuning:

    * Beautiful: `87.5%`, Nice: `100%`.
  * Post-finetuning (without freezing):

    * Generalization collapse (`0%`).
  * Post-finetuning (with layer freezing):

    * Validation accuracy restored `88-100%`.
    * Still poor test generalization due to domain shift.

### B) DTW Filtering Impact:

* DTW filtering reduced augmented video dataset size by `5x`.
* Augmented datasets without DTW: \~5 GB vs with DTW: \~850 MB.

### C) Inference Observations:

* Original vs. test video landmark mean distances:

  * Without normalization: \~30.
  * Bounding box + scale normalization: reduced to \~7.
* Procrustes alignment: reduced distance but no consistent generalization improvement.

---

## 15️⃣ Drawbacks & Challenges

* Severe **domain shift** between original dataset (DSLR shot, standing, consistent background) vs test videos (laptop camera, variable lighting, seated).
* Small dataset with **low vocabulary** (`45` labels vs `>1000` needed).
* Video sequence variation across contributors.
* Augmentation insufficient to generalize to real-world unseen videos.
* Normalization and alignment help reduce feature distance but do not fully close generalization gap.
* **Negative Finding:** ISL scalability projects are bottlenecked without a dedicated large, diverse, and consistent dataset.

---

## 16️⃣ Action Items & Next Steps

* [ ] **Write Conclusion Section** emphasizing why scalable ISL projects without sufficient datasets are a bad idea.
* [ ] Optionally, try:

  * Transfer learning with frozen lower layers.
  * Hard negative mining with out-of-domain data.
  * Test with aligned data collection environments.
* [ ] Archive best confusion matrices for supplementary.
* [ ] Export current working model for demo video generation.
* [ ] Complete **`conclusion.md`** and **`methodology.tex`** for paper.

---

## 17️⃣ Data, Code & References

* **File:** `Scalability.ipynb` for reproducibility.
* **Datasets:**

  * `holistic_labels_videos.pkl`, `rest_DTW_holistic_labels_videos.pkl`
  * Combined into `combined_labels.pkl`, `combined_landmarks.pkl`.
* **GPU:** Kaggle T4 x 2.
* **Frameworks:** TensorFlow 2.x, NumPy, scikit-learn, Matplotlib.
* **Scripts:**

  * Landmark padding/trimming to 110 frames.
  * DTW filtering.
  * Procrustes alignment implementation for frame-wise landmark alignment.

---

## 18️⃣ Metadata

* **Approximate total messages:** 900+ (massive context).
* **Word count:** \~75,000+ tokens total context.
* **Tone:** Debug-focused, advanced experimental tuning, negative finding documentation, workflow alignment for a paper.

---


