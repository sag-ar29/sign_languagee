import os
import pickle
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyttsx3
import threading

# Initialize Text-to-Speech engine in a separate thread
engine = pyttsx3.init()
def speak_text(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()

# Load model
MODEL_PATH = os.path.join('models', 'model.p')
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

labels_dict = {'0': 'Hello', '1': 'Yes', '2': 'No'}

# Initialize MediaPipe Tasks HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Translation & Smoothing State
current_sign = ""
sentence = []
hold_start_time = None
HOLD_DURATION = 1.2  # Seconds required to hold a sign before confirming
CONFIDENCE_THRESHOLD = 0.70
try: 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(mp_image)

        detected_sign = None

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks]
            y_ = [lm.y for lm in hand_landmarks]

            for lm in hand_landmarks:
                cx, cy = int(lm.x * W), int(lm.y * H)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            for lm in hand_landmarks:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1 = max(0, int(min(x_) * W) - 15)
            y1 = max(0, int(min(y_) * H) - 15)
            x2 = min(W, int(max(x_) * W) + 15)
            y2 = min(H, int(max(y_) * H) + 15)

            if len(data_aux) == 42:
                # Predict probabilities
                probs = model.predict_proba([np.asarray(data_aux)])[0]
                max_prob = np.max(probs)
                pred_class = model.classes_[np.argmax(probs)]

                # Only accept if above confidence threshold
                if max_prob >= CONFIDENCE_THRESHOLD:
                    detected_sign = labels_dict.get(str(pred_class), str(pred_class))

                    # Display bounding box and live probability
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{detected_sign} ({int(max_prob * 100)}%)",
                                (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Sentence Accumulator Logic: Confirm sign after holding pose
        if detected_sign and detected_sign == current_sign:
            if hold_start_time is None:
                hold_start_time = time.time()
            elif time.time() - hold_start_time >= HOLD_DURATION:
                # Add to sentence and speak
                if not sentence or sentence[-1] != current_sign:
                    sentence.append(current_sign)
                    speak_text(current_sign)
                hold_start_time = None
        else:
            current_sign = detected_sign
            hold_start_time = None

        # Bottom Translation Display Bar
        cv2.rectangle(frame, (0, H - 60), (W, H), (30, 30, 30), -1)
        sentence_text = " ".join(sentence[-6:])  # Show last 6 words
        cv2.putText(frame, f"Sentence: {sentence_text}", (20, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Controls hint
        cv2.putText(frame, "C: Clear | Q: Quit", (W - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Sign Language Translator', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence.clear()

finally:
    cap.release()
    cv2.destroyAllWindows()