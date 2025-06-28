import cv2
import time
import os
import threading
import speech_recognition as sr


def listen_for_commands(callback):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    print("🎤 Voice command listening started (say 'start' or 'quit').")

    while True:
        with mic as source:
            try:
                print("🎤 Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"🗣 You said: {command}")
                callback(command)
            except sr.WaitTimeoutError:
                pass  # No speech detected in time window
            except sr.UnknownValueError:
                print("❌ Didn't catch that.")
            except sr.RequestError:
                print("❌ Speech service error.")
                break

def record_video(cap, folder_name, video_count):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25  # Fallback if FPS detection fails

    target_duration = 3  # seconds
    frames_to_record = int(fps * target_duration)

    video_filename = os.path.join(folder_name, f'{folder_name}_{video_count}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    print(f"🎥 Recording: {video_filename} | Target frames: {frames_to_record} at {fps:.2f} FPS")

    recorded_frames = 0
    while recorded_frames < frames_to_record:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        recorded_frames += 1
        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("🛑 Early stop.")
            break

    out.release()
    print(f"✅ Saved: {video_filename} | Frames recorded: {recorded_frames}")

# --- Main code ---
folder_name = input("Enter the folder name to save videos: ").strip()
os.makedirs(folder_name, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

video_count = 1
running = True

def handle_command(command):
    global video_count, running
    command = command.strip()
    if command == "start":
        record_video(cap, folder_name, video_count)
        video_count += 1
    elif command == "quit":
        running = False
        print("👋 Quit command received.")
    else:
        print(f"⚠ Unrecognized command: {command}")

# Start voice listening in background thread
listener_thread = threading.Thread(target=listen_for_commands, args=(handle_command,))
listener_thread.daemon = True
listener_thread.start()

print("✅ Live feed started. Say 'start' to record or 'quit' to stop.")

while running:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Live Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quit by keypress.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Cleanup done.")