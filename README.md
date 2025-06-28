# Indian Sign Language Recognition Experiments

This repository documents **systematic experiments, pipelines, and negative findings** while building an ISL word recognition system using **landmarks and RNN architectures**, now updated with **clean Git LFS handling, structured notebooks, and scalable data workflows** for clear supervisor review and collaboration.

---

## ✨ Purpose

* Investigate ISL word classification scalability without dedicated large datasets.
* Evaluate landmark normalization, Procrustes alignment, bounding box scaling, and DTW-filtered augmentation.
* Identify limitations, challenges, and boundaries for ISL scalability.

---

## 🚀 Key Findings

* Achieved `88-92%` validation accuracy on 45-word vocabularies in controlled setups.
* Real-world data tests reveal **domain shift failures** despite advanced normalization.
* **Negative finding:** ISL scalability is bottlenecked by **data scarcity and domain inconsistency**.

---

## 🧪 Experiments Conducted

* GRU, Bi-GRU, LSTM, Bi-LSTM with various hyperparameters.
* Speed-based and DTW-based augmentations.
* Normalization: chest-centered, bounding box scaling, Procrustes.
* Real-world test video evaluations.
* Environment: Kaggle T4 x 2, 15-20 min/model.

---

## 🗂️ Repository Structure

* `Research Paper/`: All experiments, notebooks, zipped heavy data handled via **Git LFS**.
* Core notebooks:

  * `scalability.ipynb`
  * `isl-transcript-v-3-0.ipynb` ([Kaggle Link](https://www.kaggle.com/code/gautamw7/isl-transcript-v-3-0/edit))
  * `isl-transcript-v-2.0.ipynb` ([Kaggle Link](https://www.kaggle.com/code/gautamw7/isl-transcript-v-2-0?scriptVersionId=233929481))
* Supporting docs: `experiments.md`, `issues.md`, `architecture.md`.
* `figures/`: Confusion matrices, ROC curves, scatter plots.

---

## 🚫 Limitations

* Vocabulary limited to 45 words; real-world ISL uses 250+ signs.
* Persistent domain mismatch between controlled training and real-world tests.
* No external dataset integration due to constraints.

---

## 📢 Citation

> INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition
> [https://doi.org/10.1145/3394171.3413528](https://doi.org/10.1145/3394171.3413528)

---

## ✅ Conclusion

Scalable ISL word recognition using landmarks without a dedicated, diverse dataset is **impractical**.

* High validation accuracy on controlled sets does not translate to real-world robustness.
* **Data scarcity remains the core bottleneck**.
* Architectural improvements alone cannot resolve domain inconsistencies.

**Takeaway:** Prioritize dataset development and consistent recording before scalable ISL modeling.

---

## 🗺️ Future Work Roadmap

1️⃣ **Dataset Development**: Large-scale (>1000 signs), multi-signer, consistent environment dataset.
2️⃣ **Modeling**: Pretrained spatiotemporal models, transformers, multi-stream landmark fusion.
3️⃣ **Domain Adaptation**: Adversarial training, signer-adaptive embeddings.
4️⃣ **Augmentation**: Synthetic landmarks via GANs, advanced temporal strategies.
5️⃣ **Metrics Expansion**: WER, temporal consistency, beyond accuracy and AUC.

---

This structured README aligns your repo with **clean collaboration, presentation, and future scalable ISL research goals** while maintaining GitHub readability and supervisor-friendly navigation.
