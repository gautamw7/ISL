# Models to try
1.  I3D Deep Learning Architecture?
2.  SlowFast
3.  SignNet	 

# Fusion of Landmarks at different stage
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

# Hand Input (Sequential Hand Landmarks)
hand_input = Input(shape=(time_steps, 3 * 21), name="hand_input")  # 21 keypoints (x, y, z) per frame
hand_lstm = Bidirectional(LSTM(128, return_sequences=True))(hand_input)
hand_lstm = Bidirectional(LSTM(64, return_sequences=True))(hand_lstm)
hand_lstm = Bidirectional(LSTM(32))(hand_lstm)  # Final feature representation

# Face Input (Sequential Facial Landmarks)
face_input = Input(shape=(time_steps, 3 * 68), name="face_input")  # 68 keypoints (x, y, z)
face_lstm = Bidirectional(LSTM(32, return_sequences=True))(face_input)
face_lstm = Bidirectional(LSTM(16, return_sequences=True))(face_lstm)
face_lstm = Bidirectional(LSTM(8))(face_lstm)  # Final feature representation

# Shoulder Input (Sequential Shoulder Landmarks)
shoulder_input = Input(shape=(time_steps, 3 * 4), name="shoulder_input")  # 4 keypoints (x, y, z)
shoulder_lstm = Bidirectional(LSTM(8, return_sequences=True))(shoulder_input)
shoulder_lstm = Bidirectional(LSTM(4, return_sequences=True))(shoulder_lstm)
shoulder_lstm = Bidirectional(LSTM(2))(shoulder_lstm)  # Final feature representation

# Fusion Layer - Concatenating the outputs of all three branches
merged = Concatenate()([hand_lstm, face_lstm, shoulder_lstm])

# Fully Connected Layers
fc = Dense(128, activation="relu")(merged)
fc = Dropout(0.3)(fc)
fc = Dense(64, activation="relu")(fc)
fc = Dropout(0.3)(fc)
fc = Dense(32, activation="relu")(fc)

# Output Layer (Classification - Assuming 10 Sign Classes)
output = Dense(10, activation="softmax", name="output")(fc)

# Creating the Model
model = Model(inputs=[hand_input, face_input, shoulder_input], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Summary
model.summary()
