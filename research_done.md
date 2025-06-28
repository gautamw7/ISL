## Data Augmentation
How STmaps Can Help with Data Augmentation
When you augment videos for undersampled words (e.g., by applying transformations like speed variation, frame interpolation, or flipping), it's essential to ensure that the augmented videos still represent the same ISL sign. STmaps can assist in this validation process by:

Visualizing Key Features:
STmaps highlight the spatial and temporal patterns of the original video.
By comparing the STmap of the original video with that of the augmented video, you can check if the core movement patterns are preserved.
Detecting Meaning Loss:
If the augmented video's STmap shows significant deviations in movement trajectories or timing, it might indicate that the augmentation has altered the gesture in a way that changes its meaning.
For example:
If a sign requires a specific hand movement from left to right, but augmentation (e.g., flipping) reverses it, the STmap would show a mirrored trajectory, potentially changing the sign's meaning.
Quantitative Comparison:
Beyond visual inspection, you can compute similarity metrics (e.g., cosine similarity, mean squared error) between the STmaps of original and augmented videos to quantify how much the augmentation has altered the gesture.


## Landmark Extraction
2. Leverage MediaPipe’s Optimized Pipelines
Use MediaPipe Holistic: This pipeline combines pose, face, and hand tracking into a single, optimized framework. It ensures semantic consistency across detections, reducing errors caused by independent models misaligning.
Tracking Mode: MediaPipe’s tracking mode (as opposed to detection on every frame) maintains consistency across frames by relying on previous detections. This reduces errors from frame-to-frame jitter or temporary occlusions.

Fine-Tune or Retrain Models
Domain-Specific Data: If you have access to ISL-specific data, retraining MediaPipe’s detectors (e.g., the palm detector for hands or BlazePose for body pose) on this data can improve accuracy for your use case. Even a small, well-curated dataset can make a difference.
Transfer Learning: If retraining from scratch isn’t feasible, fine-tune pre-trained models on your ISL dataset to adapt them to ISL gestures.


Start with MediaPipe Holistic: It’s optimized for simultaneous detection of body, face, and hands, which is ideal for capturing ISL gestures that involve multiple body parts.
Experiment with Input Quality: Test how different video resolutions and lighting conditions affect landmark detection in your dataset.
Implement Tracking Mode: Use MediaPipe’s tracking capabilities to maintain consistency across frames, especially for fast or complex gestures.
Consider Fine-Tuning: If you have access to labeled ISL data, fine-tune the hand or pose detectors to better recognize ISL-specific gestures.
Apply Post-Processing: Use smoothing techniques to reduce noise in the detected landmarks, which can be particularly helpful for live transcription.


## Data Type:
Factor	              Method 1: Raw Frames	    Method 2: Landmark-Plotted Frames	Method 3: Landmark Arrays	Fusion     Method: Raw + Landmarks
Training Speed	      Slow (image processing)	  Moderate (simpler images)	        Fast (numerical data)	               Slowest (dual processing)
Output in Final Runs	Slow for real-time	      Moderate for real-time	          Fastest for real-time	               Moderate for real-time
Memory Utilization	  High (full frames)	      Moderate (plotted images)	        Low (arrays only)	                   Highest (frames + arrays)
Processing Power	    High (CNNs, GPUs needed)	Moderate (less complex CNNs)    	Low (RNNs or lightweight models)	   Highest (dual CNNs, powerful GPUs)
Accuracy	            High, but noise-sensitive	Moderate (loss in rendering)	    High (movement-focused)	             Highest (combined features)
Novelty	Low          (common approach)	        Low (standard CNN use)	          Moderate (optimization potential)	   High (multi-input fusion)

What I'm going to use 
Method 3 -> Method 1 -> Method 2 -> Method 4 
Depending on the GPU specs and the processing unit provided to us.

## Methodology 1: Data Auggmentation of Kaggle Dataset and Better Architecture for Model
Data Augmentation Techniques
Given your dataset's structure (words → videos → frames → landmarks), augmentation must preserve gesture semantics while increasing sample counts for underrepresented words. Suggested techniques include:

Temporal Augmentation:
Speed Variation: Adjust the sequence length by interpolating or decimating landmark sequences to simulate faster or slower signing. For example, to slow down, duplicate frames or interpolate new landmark points; to speed up, remove frames or select subsets.
Frame Interpolation/Dropping: Generate new frames between existing ones or remove frames to create variations in gesture duration, as seen in "Dataset Transformation System for Sign Language Recognition Based on Image Classification Network" (Dataset Transformation System for Sign Language Recognition Based on Image Classification Network).
Spatial Augmentation:
Translation: Shift all landmarks by a small amount (e.g., within 10% of frame size) to simulate different signer positions.
Scaling: Scale landmark coordinates to simulate varying distances from the camera, ensuring realism (e.g., not making hands too small or large).
Noise Addition: Add random Gaussian noise to landmark coordinates to simulate measurement errors or poor video quality, as mentioned in studies like "Sign Language Recognition Based on CNN and Data Augmentation".
Considerations: Avoid augmentations that alter gesture meaning, such as horizontal flipping, unless verified with an ISL expert, as direction-specific signs could be misinterpreted. Rotation around joints (e.g., wrist) might be feasible for hand landmarks but requires careful validation.
To balance your dataset, set a target number of videos per word (e.g., 20, given the maximum is 22). For words with fewer videos (e.g., 4), generate augmented videos using combinations of the above techniques until reaching the target. For example, for a word with 4 videos, generate 16 augmented videos by applying different augmentations to each original video, ensuring diversity.

Class Weighting
Cost-sensitive learning involves weighting the loss function to give more importance to minority classes. For your dataset, calculate the weight for each word as total_number_of_videos / number_of_videos_for_that_word. During training, use these weights in the loss function (e.g., CrossEntropyLoss in PyTorch) to ensure each class contributes equally to the total loss. This approach, supported by general machine learning resources like "Handling Class Imbalance in Machine Learning" (Handling Class Imbalance in Machine Learning | by Okan Yenigün | Python in Plain English), helps counteract bias toward majority classes without generating new data.

Combining Strategies
Research suggests combining data augmentation and class weighting for optimal results, as seen in "Tackling class imbalance in computer vision: a contemporary review" (Tackling class imbalance in computer vision: a contemporary review | Artificial Intelligence Review). This dual approach increases sample diversity and adjusts model focus, improving performance on underrepresented words. For instance, augment data to reach the target count, then train with class weights to fine-tune learning.

## Research Paper
1. The table with all the dataset mentioned 
2. Explain our dataset 
3. Explaining the data augmentation
4. Showing photos of data augmentation 
5. Methodology
6. Explaining and image of the model 3D image 
7. Comparing between frame and landmark extracted frame 


Here is your **`research_done.md`** ready for **Overleaf, lab notebook, or Notion integration**, capturing **all research milestones completed so far** with **dense, no-fluff, verifiable points** for your ISL negative findings paper:

---

# ✅ research\_done.md — ISL Recognition Project

## 1️⃣ Project Initialization

* Objective defined:

  * Test feasibility of **scalable ISL recognition** using landmark-based temporal models.
  * Assess **small dataset limitations** for real-world ISL deployment.
* Initial hypothesis:

  * Landmark-based GRU models with careful augmentation may generalize across users.

---

## 2️⃣ Dataset Handling

* ISL videos collected:

  * **45–262 label classes**, varying samples per label.
  * Landmark extraction using **MediaPipe**, outputting **543 keypoints (x,y,z)** → 1629 features per frame.
  * Frame padding to **110 frames** for uniform sequence lengths.
* User-recorded test dataset (2–3 sec clips) created for testing cross-domain generalization.

---

## 3️⃣ Preprocessing Work Completed

✅ **Padding:** Standardized all videos to 110 frames.
✅ **Bounding Box Normalization:** Landmark translation + scaling based on body bounding box.
✅ **Procrustes Alignment:** Applied to align landmark structures across samples for domain normalization.
✅ **DTW Augmentation:**

* Selected labels using average DTW distances.
* Augmented dataset using speed/slow/interpolation methods while filtering with DTW thresholds.
* Reduced augmented video size by \~5x while maintaining similarity.

---

## 4️⃣ Modeling Experiments Completed

✅ Built **Bidirectional GRU-based architecture** for temporal ISL landmark classification.
✅ Extensive hyperparameter sweeps:

* Learning rates: `1e-3`, `1e-4`, `1e-5`.
* Dropout configs: `(0, 0.1, 0.1)` and `(0, 0.1, 0.15)`.
* Early stopping patience: 10, 12.
* Batch size: 16.
  ✅ Training and evaluation on 45-label dataset:
* Final training accuracy: **98.9%**.
* Final validation accuracy: **92.3%**.
* F1-score: **0.9208**, AUC-ROC: **0.9985**.
  ✅ Visual validation:
* Scatter plots, heatmaps, ROC curves, confusion matrices generated for thorough validation.

---

## 5️⃣ Testing & Fine-Tuning Completed

✅ **Testing on user-recorded videos**:

* Detected significant domain mismatch even after normalization/alignment.
* Mean landmark distance still non-trivial, limiting generalization.

✅ **Fine-tuning Experiments:**

* Fine-tuning without freezing → catastrophic forgetting, 0% accuracy on prior labels.
* Fine-tuning with layer freezing:

  * Achieved 100% validation on small samples but still 0% on prior dataset.
  * Overfitting on few-shot samples, unable to generalize.
    ✅ Validation on test labels:
* Certain labels ("Beautiful", "Nice") evaluated post-fine-tuning.
* Consistent misclassification post-fine-tuning noted.

---

## 6️⃣ Key Findings

✅ Landmark-based GRU models perform well within the dataset but fail to generalize across domains.
✅ Procrustes and bounding box normalization improve structure alignment but do not solve environment-based variation.
✅ DTW-based augmentation stabilizes training and label consistency within the dataset.
✅ Fine-tuning on small samples without large diverse data leads to **catastrophic forgetting**.
✅ **Scalable ISL recognition requires large, diverse, well-labeled datasets**—not feasible with small isolated datasets.

---

## 7️⃣ Documentation & Tracking Completed

✅ **Architecture.md** (updated to current pipeline).
✅ **Experiments.md** (full experiment logs).
✅ Heatmaps, scatter plots, confusion matrices archived for figures in the paper.
✅ Detailed logs of testing runs, model metrics, and debug outputs tracked systematically.

---

## 8️⃣ Paper Positioning Finalized

✅ Will publish as **“Negative Findings on Small-Scale ISL Recognition with Landmark-Based Models”**.
✅ Framed around **pitfalls, limitations, and domain shift challenges** for ISL.
✅ Highlights the **necessity of large-scale datasets** for ISL recognition pipelines.
✅ Ready for **IEEE/CVPR/NeurIPS workshops focused on Sign Language, Responsible AI, or ML Failures**.

---

## 9️⃣ Technical Tools Used

* **TensorFlow/Keras:** Modeling and training.
* **NumPy, Pandas:** Data handling.
* **scikit-learn:** Metrics and confusion matrices.
* **Matplotlib, Seaborn:** Visualization.
* **OpenCV, MoviePy:** Frame extraction and video handling.
* **MediaPipe:** Landmark extraction.
* Optional integration with `wandb` (not fully used here).

---

## 10️⃣ Pending / Optional Extensions

* [ ] Prepare paper draft (`paper.md` / Overleaf integration).
* [ ] Generate formal BibTeX references for datasets/tools.
* [ ] (Optional) Archive final code and dataset splits for reproducibility.
* [ ] (Optional) Attempt minimal multi-person testing for robustness discussion.
* [ ] (Optional) Run ablation with Procrustes + bounding box + DTW on a few additional labels for figure clarity.

---

## 11️⃣ Metadata

* **Current phase:** Wrapping up experiments and documentation for paper submission.
* **Total messages analyzed:** \~250+.
* **Total words exchanged:** \~60,000+.
* **Tone:** Debug-heavy, iterative, critical workflow refinement for negative result positioning.

---
