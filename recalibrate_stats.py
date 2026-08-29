import numpy as np
import os

BASE_PATH = os.getcwd()
MODELS_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("[INFO] Recalibrage des statistiques de normalisation...")

try:
    X = np.load("X_cnn.npy")
    print(f"[OK] Fichier X_cnn.npy chargé. Shape: {X.shape}")
    
    X_mean = X.mean()
    X_std  = X.std()
    
    np.save(os.path.join(MODELS_DIR, 'norm_mean.npy'), X_mean)
    np.save(os.path.join(MODELS_DIR, 'norm_std.npy'), X_std)
    
    print(f"[SUCCESS] Stats sauvegardées !")
    print(f"Moyenne : {X_mean}")
    print(f"Ecart-type : {X_std}")
    print("Ton interface Streamlit utilisera désormais ces valeurs compatibles avec le modèle à 91%.")
    
except Exception as e:
    print(f"[ERREUR] Impossible de charger X_cnn.npy : {e}")
