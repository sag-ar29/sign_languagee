import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join('data', 'processed', 'data.pickle')
MODEL_PATH = os.path.join('models', 'model.p')

# Load dataset
with open(DATA_PATH, 'rb') as f:
    dataset = pickle.load(f)

data = np.asarray(dataset['data'])
labels = np.asarray(dataset['labels'])

# Split into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize and train classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# Evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"Model trained successfully. Test Accuracy: {score * 100:.2f}%")

# Save the trained model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Model saved to {MODEL_PATH}")