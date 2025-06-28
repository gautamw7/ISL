# ISL

# Indian Sign Language Recognition Experiments

This repository documents **systematic experiments, pipelines, and negative findings** while building an ISL word recognition system using **landmarks and RNN architectures**.

## ✨ Purpose

- Investigate scalability of ISL word classification without a dedicated large dataset.
- Evaluate landmark normalization, Procrustes alignment, bounding box scaling, and DTW-filtered augmentation.
- Identify limitations, challenges, and realistic boundaries for ISL scalability research.

## ⚡ Key Findings

- Validation accuracies of `88-92%` were achieved on 45-word vocabularies in a controlled dataset.
- Real-world test data showed **domain shift failure**, even after fine-tuning and advanced normalization.
- **Negative finding**: ISL word recognition is bottlenecked by **data scarcity** and **domain inconsistencies**, making scalable models impractical without significant data expansion.

## 🧪 Experiments

- GRU, Bi-GRU, LSTM, Bi-LSTM models with various dropouts and learning rates.
- Augmentation with speed changes and DTW-based filtering.
- Normalization experiments:
  - Chest-centered.
  - Bounding box position + scale.
  - Procrustes alignment.
- Real-world test videos captured to assess generalization.
- GPU: Kaggle T4 x 2, typical training time `15-20 min/model`.

## 🗂️ Repository Structure

- `experiments.md` – Full experimental log.
- `issues.md` – Bottlenecks and challenges.
- `architecture.md` – Final model architecture.
- `scalability.ipynb` – Core experimentation notebook.
- `isl-transcript-v-3-0.ipynb` – Pipeline and final notebook before experiments notebook - https://www.kaggle.com/code/gautamw7/isl-transcript-v-3-0/edit
- `isl-transcript-v-2.0.ipynb` – Pipeline and intial testing - https://www.kaggle.com/code/gautamw7/isl-transcript-v-2-0?scriptVersionId=233929481
- `figures/` – Confusion matrices, ROC curves, and scatter plots for publication.

## 🚫 Limitations

- Small vocabulary (`45 words`) insufficient for real-world ISL scalability.
- Domain mismatch between train/test environments remains unresolved.
- No external dataset integration due to constraints.

## 📢 Citation
title = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
url = {https://doi.org/10.1145/3394171.3413528},


# Conclusion

Through systematic experimentation, this study demonstrates that **scalable Indian Sign Language word recognition using landmarks without a dedicated, diverse, and large dataset is impractical**.

Key conclusions:

- Even with advanced normalization (bounding box scaling, Procrustes alignment), models fail to generalize to real-world test environments.
- High validation accuracies (`~92%`) on controlled datasets create a false sense of performance when tested on varied camera settings, lighting, and signer conditions.
- Data scarcity remains the fundamental bottleneck, overshadowing architectural or augmentation improvements.
- **Negative finding:** It is critical to prioritize dataset development and environment-consistent recording before attempting scalable ISL modeling efforts.

This paper highlights the **need for dedicated dataset curation and domain-aligned data pipelines** as prerequisites for practical, deployable ISL recognition systems.

# Future Work

Based on the findings from this research, the following actionable steps are recommended:

1️⃣ **Dataset Development**
- Create a large-scale ISL dataset covering `>1000` signs with:
  - Multiple signers of varying demographics.
  - Multiple camera settings and environments.
  - Consistent lighting, angle, and framing.

2️⃣ **Architecture Enhancements**
- Integrate pretrained spatiotemporal models (I3D, SlowFast, SignNet).
- Experiment with Transformer or CTC-based models for continuous recognition.
- Use multi-stream landmark fusion (hands, face, shoulders) for richer context.

3️⃣ **Domain Adaptation**
- Employ adversarial training to align feature distributions across domains.
- Explore signer-adaptive embeddings to improve generalization.

4️⃣ **Data Augmentation**
- Investigate synthetic landmark generation using GANs or simulation environments.
- Utilize advanced temporal augmentation strategies to increase variability.

5️⃣ **Metric Expansion**
- Beyond accuracy and AUC, compute WER (Word Error Rate) and temporal consistency checks to assess practical viability.

This future roadmap can help transition from controlled lab experiments toward practical, scalable ISL recognition systems capable of real-world deployment.


