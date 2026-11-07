import os
import cv2

DATA_DIR = os.path.join('data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

# Number of sign classes and images per sign
NUMBER_OF_CLASSES = 3       # Start with 3 signs (e.g., 0: 'Hello', 1: 'Yes', 2: 'No')
DATASET_SIZE = 100          # 100 frames per class

cap = cv2.VideoCapture(0)

for j in range(NUMBER_OF_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_dir, exist_ok=True)

    print(f'Collecting data for class {j}')

    # Wait for the user to be ready
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Class {j}: Press "Q" to start capture', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collector', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images
    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        cv2.imshow('Collector', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()