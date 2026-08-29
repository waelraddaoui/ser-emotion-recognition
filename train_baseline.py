import os
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
BASE_PATH = os.getcwd() # Use relative to avoid path issues
X_PATH = os.path.join(BASE_PATH, 'X_cnn.npy')
Y_PATH = os.path.join(BASE_PATH, 'y_labels.npy')
MODELS_DIR = os.path.join(BASE_PATH, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("--- Training MLP Baseline ---")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # ── Sécurité : filtrer les labels hors-plage ───────────────────────────
    N_CLASSES = 8
    valid_mask = (y >= 0) & (y < N_CLASSES)
    if not np.all(valid_mask):
        print(f"  ⚠ {np.sum(~valid_mask)} échantillons invalides supprimés.")
        X = X[valid_mask]
        y = y[valid_mask]
    print(f"Dataset : {X.shape[0]} échantillons, {N_CLASSES} classes")

    from sklearn.preprocessing import StandardScaler

    # Aplatir les features 2D pour le MLP
    X_flat = X.reshape(X.shape[0], -1)

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y)

    print("Fitting MLP...")
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=200,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        learning_rate='adaptive',
        learning_rate_init=1e-3,
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"MLP Baseline Accuracy: {acc:.4f}")
    
    # Save model and scaler
    with open(os.path.join(MODELS_DIR, 'mlp_baseline.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, 'mlp_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Baseline Done.")

if __name__ == "__main__":
    main()
