# 📄 ISL Recognition Architecture (Updated)

## 1️⃣ Models Considered Previously

* I3D Deep Learning Architecture (not used due to small dataset constraints).
* SlowFast (considered but skipped due to video-heavy GPU demand).
* SignNet (explored conceptually, but moved to landmark + GRU pipeline for interpretability).

---

## 2️⃣ Current Effective Pipeline

**🧩 Pipeline Rationale:**

* ISL dataset with 45–262 labels.
* Landmark extraction (`MediaPipe` or similar).
* Frame-wise **padding** to uniform sequence length (`110` frames).
* **Bounding box normalization** + **Procrustes alignment** for domain normalization.
* **DTW-based augmentation** to stabilize temporal variations.
* GRU-based architecture for temporal modeling.
* Softmax classification on the final layer.

---

## 3️⃣ Current Model Architecture

```python
inputs = Input(shape=(110, 1629))

# First GRU block
x = GRU(384, return_sequences=True)(inputs)
x = LayerNormalization()(x)
x = Dropout(0)(x)

# Second GRU block
x = GRU(256, return_sequences=True)(x)
x = LayerNormalization()(x)
x = Dropout(0.1)(x)

# Third GRU block
x = GRU(128, return_sequences=True)(x)
x = LayerNormalization()(x)

# Attention block
def attention_block(x):
    attention_scores = Dense(1, activation='tanh')(x)
    attention_weights = Lambda(lambda s: tf.nn.softmax(s, axis=1))(attention_scores)
    weighted_sum = Lambda(lambda s: tf.reduce_sum(s[0] * s[1], axis=1))([x, attention_weights])
    return weighted_sum

x = attention_block(x)

# Final dense + output
x = Dense(64, activation='relu')(x)
x = Dropout(0.1)(x)

outputs = Dense(num_classess, activation='softmax')(x)

model = Model(inputs, outputs)

```

---

## 4️⃣ Training Strategy

* **Batch size:** 16
* **Epochs:** 80 (baseline), 30–70 (fine-tuning).
* **Callbacks:**

  * `EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)`
  * `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)`
* **Validation split:** typically 20%.
* **Loss:** categorical cross-entropy.
* **Metrics:** accuracy, F1, precision, recall, AUC-ROC tracked externally.
* **Visualization:** heatmaps, scatter plots for landmark debugging.

---

## 5️⃣ Fine-Tuning Workflow

* Load pre-trained model weights from the baseline trained on 45 classes.
* Freeze lower GRU layers (optional).
* Recompile with a lower learning rate (`1e-4` or `5e-5`).
* Fine-tune on user-recorded videos for specific labels (e.g., "Beautiful", "Nice").
* Evaluate generalization post-fine-tuning (observe potential catastrophic forgetting).

---

## 6️⃣ Notes & Future Considerations

✅ Model architecture is **lightweight and scalable** for ISL recognition if **data diversity issues are resolved**.
✅ Procrustes alignment improves domain alignment but does not fully solve generalization issues.
✅ DTW augmentation stabilizes training under small dataset conditions.
✅ Currently focusing on **documenting negative findings** to demonstrate the **limitations of low-resource ISL systems** for IEEE paper submission.

---

## 7️⃣ Additional Tools Integrated

* `scikit-learn`: for confusion matrices, classification reports.
* `matplotlib`, `seaborn`: for ROC curves, heatmaps, scatter plots.
* Optional `wandb` or `TensorBoard` for hyperparameter tuning (not explicitly used in the current experiments).

---

## 8️⃣ Version Tracking

* **Last update:** 2025-06-28
* **Owner:** \ Gautam
* **Project:** ISL Recognition Negative Results Documentation

---

If you need, I can also generate:

✅ **`model_finetuning.py`** script matching this architecture with layer-freezing logic ready for your experiments.
✅ **Overleaf/Notion insertion-ready architecture diagrams** reflecting this structure.
✅ A **short summary section for your paper’s methods section** using this exact architecture.

Let me know if you want these next for smooth workflow integration.
