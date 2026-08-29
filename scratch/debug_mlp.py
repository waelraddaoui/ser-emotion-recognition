import pickle
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X = np.load('X_cnn.npy')
y = np.load('y_labels.npy')
X_flat = X.reshape(X.shape[0], -1)

models_dir = 'models'
mlp_path = os.path.join(models_dir, 'mlp_baseline.pkl')

EMOTION_NAMES = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful"]

if os.path.exists(mlp_path):
    with open(mlp_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check predictions on the whole set
    y_pred = model.predict(X_flat)
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=EMOTION_NAMES))
    
    # Check for bias
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\nPrediction distribution:")
    print(dict(zip(unique, counts)))
    
    # Check one sample scaling
    print(f"\nFeature mean: {np.mean(X_flat)}")
    print(f"Feature std: {np.std(X_flat)}")
    print(f"Feature range: {np.min(X_flat)} to {np.max(X_flat)}")
else:
    print(f"Model not found at {mlp_path}")
