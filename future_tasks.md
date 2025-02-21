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

Frame Interpolation:
Option[1]: Data Augmentation
Generate intermediate frames between existing frames to create new variations of a video.
Tools like OpenCV or ffmpeg can help with frame interpolation.
Time Stretching/Speed Variation:
Slightly slow down or speed up videos (e.g., 0.9x or 1.1x speed) to create variations.
This preserves the meaning of the gesture while adding diversity.
Horizontal Flipping:
Flip the video horizontally to create a mirrored version.
Important: Check with an ISL expert to ensure flipping doesn't change the meaning of the gesture, as some gestures might be direction-specific.
Cropping/Zooming:
Apply minor crops or zooms to focus on different parts of the gesture, creating slight variations.
Noise Addition:
Add minor noise to the landmarks (e.g., slightly shifting key points) to simulate natural variations in gestures.
By applying these techniques, you can generate additional videos for words with fewer than the target number (e.g., 20 videos per word).

Option 2: Synthetic Video Generation (Advanced)
Use advanced techniques like GANs (Generative Adversarial Networks) to generate new videos for underrepresented words.
This approach is complex and resource-intensive, so it's recommended only if simpler augmentation techniques don't suffice.

Option 3: Undersampling (Optional)
If some words have significantly more videos than others, you could reduce the number of videos for overrepresented words (e.g., randomly remove videos).
However, since your maximum video count is 22, undersampling might not be necessary unless you want to strictly enforce a lower target (e.g., 16 videos per word). Be cautious, as undersampling risks losing valuable data.

6. Handling Varying Sequence Lengths
Since videos have different lengths (i.e., different numbers of frames), you need a consistent input size for your model. Here's how to handle this:

Fixed Sequence Length:
Choose a target sequence length based on your dataset. For example, you could use the average number of frames across all videos (e.g., ~16 frames).
For shorter videos:
Pad the sequences with zeros to match the target length.
Alternatively, repeat frames to fill the gap.
For longer videos:
Truncate the sequences by cutting off excess frames.
Alternatively, sample key frames (e.g., select every nth frame) to reduce the length without losing critical information.
Dynamic Handling (Advanced):
Some models, like LSTMs, can handle variable-length sequences. However, this complicates batching during training, so it's often easier to standardize sequence lengths.

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
Step 2: Balance the Dataset
Target Video Count:
Aim for a balanced number of videos per word. Since the maximum is 22 and the average is ~16, you could target around 20 videos per word.
Augmentation for Underrepresented Words:
For words with fewer than 20 videos, apply data augmentation techniques:
Speed Variation: Create slowed-down or sped-up versions (e.g., 0.9x and 1.1x speed).
Frame Interpolation: Use tools like OpenCV or ffmpeg to interpolate frames.
Horizontal Flipping: If applicable (verify with an ISL expert).
Aim to generate enough augmented videos to reach ~20 per word.
Undersampling for Overrepresented Words (Optional):
If any words have significantly more than 22 videos, you could randomly undersample them. However, since your max is 22, this might not be necessary.
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

8. Revisiting SMOTE
As discussed earlier, SMOTE isn't suitable for your dataset because it isn't designed for sequential data like video landmark sequences. Instead of SMOTE, rely on the augmentation techniques mentioned above.

If you still face severe imbalance after augmentation, consider:

Class Weighting:
During training, assign higher weights to underrepresented classes to make the model pay more attention to them.
Focal Loss:
Use a loss function that focuses on hard-to-classify examples, which can help address imbalance.

