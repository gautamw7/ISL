# 🛠️ Issues.md

---
## 1️⃣ Dataset & Data Quality Issues

* **Limited Vocabulary:**

  * Current dataset covers only `45` ISL labels.
  * Human signers typically have vocabularies of `>200` words; robust machine sign recognition requires `>1000` to generalize.
* **Domain Shift:**

  * Training data recorded on DSLR, standing, consistent backgrounds.
  * Test data captured on a laptop camera, often seated, with different lighting, angle, and background, leading to severe distribution mismatch.
* **Insufficient Diversity:**

  * Single-person or small group data collection lacks signer diversity (height, body shape, skin tone, signing styles).
* **Video Length Disparities:**

  * Training videos: \~3 seconds, 110 frames.
  * Test videos: often <2 seconds, 50–60 frames.
  * This mismatch created alignment difficulties even with padding/interpolation.
* **Augmentation Limitations:**

  * Speed variations (slow/fast), interpolation, and DTW-based filtering reduce size but fail to introduce sufficient variation for generalization.

---

## 2️⃣ Preprocessing & Landmark Normalization Issues

* **Normalization Conflicts:**

  * Chest-centered normalization reduced positional variance but did not fully handle body scale variance across signers.
  * Bounding box normalization helped reduce scale issues but did not solve generalization gaps.
  * Procrustes alignment reduced mean landmark distance but led to inconsistent performance and sometimes degraded model accuracy.
* **Lack of Depth Information:**

  * Landmarks are limited by the accuracy of MediaPipe holistic, with less reliability in `z` coordinates, especially with different camera qualities.
* **Sequence Misalignment:**

  * Despite trimming/padding, temporal misalignment remained for short videos, impacting GRU/LSTM temporal modeling.

---

## 3️⃣ Model & Training Issues

* **Overfitting:**

  * High train accuracy (`>98%`) with validation accuracy capping (`~88-92%`), indicating overfitting on limited label sets.
* **Fine-tuning Failure:**

  * Direct fine-tuning on new labels without layer freezing led to catastrophic forgetting (`0%` test accuracy).
  * Even with layer freezing, the test set accuracy remained low if the domain was not aligned.
* **Limited Hyperparameter Exploration:**

  * Due to GPU constraints, extensive sweeps over optimizer variations, learning rates, and architectures were not performed.
* **Lack of Transfer Learning:**

  * No backbone pre-trained model (I3D, SlowFast, or pretrained landmark embeddings) used, which could improve generalization but was constrained by dataset limitations.

---

## 4️⃣ Hardware & Environment Issues

* **GPU Constraints:**

  * Kaggle T4 x 2 limited to 12-hour sessions; long experiments required checkpointing and resumption.
* **File System Limitations:**

  * Repeated file uploads/downloads due to ephemeral environments.
* **RAM/VRAM Limits:**

  * Loading full 45-label datasets with multiple augmentations sometimes caused memory overflow.

---

## 5️⃣ Experimental Findings (Negative)

* **Negative Finding:**

  * It is impractical to scale ISL word recognition reliably without:

    * A large, diverse dataset.
    * Controlled recording environments for train/test alignment.
    * Integration of signer adaptation or robust domain adaptation techniques.
* **False Sense of High Validation Accuracy:**

  * High validation scores on controlled data did not translate to real-world test data generalization.
* **Preprocessing Cannot Fix Data Scarcity:**

  * Procrustes alignment, bounding box scaling, or normalization improved intra-domain consistency but could not solve inter-domain mismatch due to insufficient variation in training data.

---

## 6️⃣ Pending & Unresolved Issues

* **Lack of External Dataset Integration:**

  * Did not test or align with existing ISL datasets due to label mismatch and constraints.
* **No Cross-signer Testing:**

  * Model untested across multiple signers, limiting claims of generalization.
* **Fusion Models Unexplored:**

  * Multi-stream landmark fusion architecture planned but not executed due to time constraints.
* **Metric Diversity:**

  * Primarily tracked accuracy, F1, AUC; sign recognition often benefits from WER (Word Error Rate) and temporal consistency checks, which were not computed.

---

## 7️⃣ Future Considerations

* Build or collect a **clean, diverse, large ISL dataset** covering `>1000` signs.
* Integrate **domain adaptation** or **signer-invariant embeddings**.
* Use **pre-trained action recognition backbones** (I3D, SlowFast) for temporal modeling.
* Explore **CTC loss or Transformer architectures** for continuous sign recognition.
* Revisit **augmentation pipelines** with synthetic hand/pose rendering or GANs for data enrichment.

---

