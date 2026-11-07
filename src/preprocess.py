import os
import pickle
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

# Auto-download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = os.path.join('data', 'raw')
OUTPUT_PATH = os.path.join('data', 'processed', 'data.pickle')

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {dir_}")
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)

        # Process if at least one hand is detected
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks]
            y_ = [lm.y for lm in hand_landmarks]

            # Normalize coordinates relative to min x and y
            for lm in hand_landmarks:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save features and labels
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Preprocessing complete! Total valid samples extracted: {len(data)}")