Research Focus:
Papers on sign language recognition (CNN-RNN, transfer learning).
Techniques for imbalanced video data.

Step 1: Explore and Combine Datasets
Kaggle Dataset: You have 1780 words with varying video counts (4 to 22 videos per word).
Additional ISL Datasets:
Investigate other ISL datasets you've found to see if they can supplement your Kaggle dataset.
Check for:
Overlap: Do they cover the same or additional words?
Quality: Are the video resolution, frame rates, and landmark extractability similar?
Annotations: Do they have text labels for the gestures? 


22 Feb 2025 1.10 AM
Biased Model: The model may perform better on words with more videos and worse on words with fewer videos because it has more examples to learn from for certain words.
Poor Generalization: Words with fewer videos might not be learned effectively, leading to lower accuracy for those words.
To address this, you need to balance the dataset. Balancing can be done by either:

Increasing the number of examples for underrepresented words (e.g., words with fewer videos).
Reducing the number of examples for overrepresented words (e.g., words with more videos, though this is less common in your case since the max is 22).

22 Feb 2025 1.15 AM
Option 1: Data Augmentation
Data augmentation involves applying transformations to existing videos to create new, realistic samples for underrepresented words. Here are some augmentation techniques you can use:

7. Step-by-Step Solution for Dataset Handling
Let's outline a concrete plan to address your dataset challenges:

Step 1: Explore and Combine Datasets
Kaggle Dataset: You have 1780 words with varying video counts (4 to 22 videos per word).
Additional ISL Datasets:
Investigate other ISL datasets you've found to see if they can supplement your Kaggle dataset.
Check for:
Overlap: Do they cover the same or additional words?
Quality: Are the video resolution, frame rates, and landmark extractability similar?
Annotations: Do they have text labels for the gestures?
If compatible, merge the additional datasets with your Kaggle dataset to increase the number of videos per word.


Step 3: Preprocess the Videos
Landmark Extraction:
Use a reliable landmark extraction tool like MediaPipe or OpenPose to get consistent landmarks across frames.
Ensure that the same set of landmarks is extracted for each frame (e.g., hand, face, and body keypoints).
Sequence Length Standardization:
Determine a Fixed Length: Calculate the average number of frames across all videos and use that as your target sequence length.
Padding/Truncation:
For videos with fewer frames, pad the sequences with zeros.
For longer videos, truncate them or sample frames uniformly.
Alternative: Use key frame selection or temporal downsampling to reduce longer sequences without losing critical information.
Step 4: Prepare the Data for Training
Data Format:
Each video should be represented as a sequence of landmark vectors, e.g., a 3D array of shape (sequence_length, num_landmarks, 3) if using 3D coordinates.
Labeling:
Ensure each sequence is labeled with the corresponding word.
Train-Test Split:
Split your data into training and validation sets (e.g., 80-20 split), ensuring that the split is stratified (i.e., maintains the distribution of words).

1 March 2025:
1. Downloading and checking the following data sources:
   Datasets
https://data.mendeley.com/datasets/kcmpdxky7p/1?utm_source=chatgpt.com
https://paperswithcode.com/dataset/fdmse-isl?utm_source=chatgpt.com
https://huggingface.co/datasets/Exploration-Lab/CISLR?utm_source=chatgpt.com
https://paperswithcode.com/dataset/isltranslate
https://aclanthology.org/2022.emnlp-main.707/
https://data.mendeley.com/datasets/kcmpdxky7p/1

Name
CISLR
ISL - Include

Maybe Testing Dataset 
https://osf.io/gn8k5/files/osfstorage

Helpful Data
https://data.mendeley.com/datasets/2vfdm42337/1

Could be use
https://www.kaggle.com/datasets/harsh0239/isl-indian-sign-language-video-dataset

2. Normalization should be done or not
3. Best type of split for training
4. Best type of model for training and why
5. What type of NLP to use
6. What will be the final method to get text from the video 
