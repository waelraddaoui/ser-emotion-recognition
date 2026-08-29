import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONUTF8"] = "1"

MODELS_DIR = os.path.join(os.getcwd(), 'models')
EMOTION_NAMES_ALL = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

def main():
    print("--- Evaluating CNN Model (Metrics Only) ---")
    
    # Load model
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'best_model.keras'))
    
    # Load test data
    X_test = np.load(os.path.join(MODELS_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(MODELS_DIR, 'y_test.npy'))
    
    # Detect actual classes present in the test set
    present_classes = sorted(set(y_test.tolist()))
    target_names = [EMOTION_NAMES_ALL[i] for i in present_classes if i < len(EMOTION_NAMES_ALL)]
    print(f"  Classes detectees dans y_test : {present_classes} -> {target_names}")

    # Predict
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=present_classes, target_names=target_names, zero_division=0)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save report
    with open('eval_report_simple.txt', 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(report)
    print("Done.")

if __name__ == "__main__":
    main()
