"""
=============================================================================
prepare_data.py — Extraction des Features Audio pour CNN (RAVDESS)
=============================================================================
Titre du PFE : "Classification des Émotions Vocales à partir de MFCCs
               à l'aide de Réseaux Neuronaux Convolutifs : 
               Une Étude sur le Corpus RAVDESS"

Ce script extrait des features MFCC 2D (+ Delta + Delta²) depuis chaque
fichier audio WAV du corpus RAVDESS, avec augmentation de données.

Sortie :
  - X_cnn.npy   : shape (N, 40, 216, 1)  — features 2D pour CNN
  - y_labels.npy : shape (N,)             — labels d'émotion (0-7)
  - label_map.pkl                         — mapping émotion → index
=============================================================================
"""

import os
import argparse
import pickle
import numpy as np
import librosa
import warnings
from tqdm import tqdm
import json
import sys

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_PATH    = os.getcwd()
DATA_PATH    = os.path.join(BASE_PATH, "data")
OUTPUT_PATH  = BASE_PATH

SAMPLE_RATE  = 22050     # Hz
DURATION     = 2.5       # secondes
OFFSET       = 0.5       # secondes
N_MFCC       = 40        # nombre de coefficients MFCC
MAX_FRAMES   = 216       # frames temporelles fixes (padding / truncation)

# Augmentation : activer pour agrandir le dataset
AUGMENT      = True

# Mapping RAVDESS : 3ème segment du nom de fichier → émotion
# NOTE : On utilise TOUTES les 8 émotions (codes 1-8)
EMOTION_MAP = {
    1: "Neutral",
    2: "Calm",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgust",
    8: "Surprised",
}

# Labels zero-indexed (0-7)
LABEL_MAP = {k: k-1 for k in EMOTION_MAP.keys()}

# Codes valides (Toutes les 8 émotions du corpus)
VALID_EMOTION_CODES = {1, 2, 3, 4, 5, 6, 7, 8}
ADMIN_CURATED_DIR = "admin_curated"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS D'AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def add_noise(y, noise_factor=0.005):
    """Ajoute du bruit gaussien pour simuler un environnement bruité."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def time_stretch(y, rate=None):
    """Étire ou compresse le signal dans le temps."""
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y, sr, steps=None):
    """Décale la hauteur tonale du signal."""
    if steps is None:
        steps = np.random.randint(-3, 4)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DES FEATURES MFCC 2D
# ─────────────────────────────────────────────────────────────────────────────

def extract_mfcc_2d(y, sr, n_mfcc=N_MFCC, max_frames=MAX_FRAMES):
    """
    Extrait et concatène MFCC + Delta + Delta² pour former une matrice 2D.
    
    Retourne une matrice de shape (n_mfcc * 3, max_frames) = (120, 216).
    """
    # MFCC de base
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Dérivée première (delta) — capture la dynamique temporelle
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Dérivée seconde (delta-delta) — capture l'accélération
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Concaténation des 3 représentations
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    # shape: (n_mfcc*3, T) = (120, T)

    # Padding ou truncation pour avoir une longueur fixe
    if combined.shape[1] < max_frames:
        pad_width = max_frames - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :max_frames]

    return combined  # shape: (120, max_frames)


# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE D'EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def process_file_raw(file_path):
    """
    Charge et prétraite le signal brut de manière robuste.
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_norm = librosa.util.normalize(y_trimmed)
    y_filt = librosa.effects.preemphasis(y_norm)
    
    target_len = int(DURATION * sr)
    if len(y_filt) > target_len:
        y_final = y_filt[:target_len]
    else:
        y_final = librosa.util.pad_center(data=y_filt, size=target_len)
    
    return y_final, sr


def collect_ravdess_files():
    files = []
    for actor_dir in sorted(os.listdir(DATA_PATH)):
        actor_path = os.path.join(DATA_PATH, actor_dir)
        if os.path.isdir(actor_path) and actor_dir.startswith("Actor_"):
            for fname in os.listdir(actor_path):
                if fname.endswith('.wav'):
                    parts = fname.split('-')
                    if len(parts) >= 7:
                        emotion_code = int(parts[2])
                        if emotion_code in VALID_EMOTION_CODES:
                            files.append((os.path.join(actor_path, fname), LABEL_MAP[emotion_code]))
    return files


def collect_admin_curated_files():
    curated_path = os.path.join(DATA_PATH, ADMIN_CURATED_DIR)
    if not os.path.isdir(curated_path):
        return []

    emotion_to_label = {name: code - 1 for code, name in EMOTION_MAP.items()}
    files = []
    for emotion_name, label in emotion_to_label.items():
        emotion_dir = os.path.join(curated_path, emotion_name)
        if not os.path.isdir(emotion_dir):
            continue
        for fname in sorted(os.listdir(emotion_dir)):
            if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
                files.append((os.path.join(emotion_dir, fname), label))
    return files


def main():
    all_features = []
    all_labels   = []

    parser = argparse.ArgumentParser(description="Prepare dataset features")
    parser.add_argument("--admin-only", action="store_true", help="Use only admin curated audio files (ignore RAVDESS)")
    parser.add_argument("--new-only", action="store_true", help="Use only newly collected admin curated files not seen in previous trainings")
    parser.add_argument("--output-prefix", default="", help="Prefix output .npy files, e.g. admin_feedback")
    args = parser.parse_args()

    # manifest to track curated files already used in previous trainings
    curated_manifest_path = os.path.join(DATA_PATH, ADMIN_CURATED_DIR, ".used_manifest.json")
    used_set = set()
    if args.new_only:
        if os.path.exists(curated_manifest_path):
            try:
                with open(curated_manifest_path, 'r', encoding='utf-8') as mf:
                    used = json.load(mf)
                    if isinstance(used, list):
                        used_set = set(used)
            except Exception:
                used_set = set()

    # Lister les fichiers supervises: RAVDESS + sons reels valides par l'admin.
    ravdess_files = collect_ravdess_files()
    curated_files = collect_admin_curated_files()
    if args.new_only:
        # filter curated_files to only those not in manifest
        curated_root = os.path.join(DATA_PATH, ADMIN_CURATED_DIR)
        new_curated = []
        for path, label in curated_files:
            rel = os.path.relpath(path, curated_root)
            if rel not in used_set:
                new_curated.append((path, label))
        curated_files = new_curated

    if args.admin_only or args.new_only:
        all_files = curated_files
    else:
        all_files = ravdess_files + curated_files

    print(f"\n{'='*60}")
    print(f"  PFE — Classification des Émotions Vocales (RAVDESS)")
    print(f"  Fichiers RAVDESS : {len(ravdess_files)}")
    print(f"  Fichiers admin curated : {len(curated_files)}")
    if args.admin_only:
        print("  Mode: TRAIN ON ADMIN-CURATED ONLY")
    if args.new_only:
        print("  Mode: TRAIN ON NEWLY-COLLECTED CURATED FILES ONLY")
    print(f"  Nombre de fichiers trouvés : {len(all_files)}")
    print(f"  Features : MFCC + Delta + Delta2 -> (120, {MAX_FRAMES})")
    print(f"  Augmentation : {'Activee' if AUGMENT else 'Desactivee'}")
    print(f"{'='*60}\n")

    if len(all_files) == 0:
        if args.new_only:
            print("=== PREPARE_DATA: NO_NEW_CURATED_FILES ===")
            sys.exit(0)
        print("Erreur : aucun fichier audio trouve pour preparer les donnees.")
        sys.exit(1)

    for file_path, label in tqdm(all_files, desc="Extraction MFCC 2D"):
        try:
            # 1. Prétraitement Robuste (Trim + Normalize + Preemphasis)
            y_proc, sr = process_file_raw(file_path)
            
            # 2. Version originale (Prétraitée)
            all_features.append(extract_mfcc_2d(y_proc, sr))
            all_labels.append(label)

            if AUGMENT:
                # 3. Bruit gaussien
                y_noisy = add_noise(y_proc)
                all_features.append(extract_mfcc_2d(y_noisy, sr))
                all_labels.append(label)

                # 4. Pitch shift
                y_pitched = pitch_shift(y_proc, sr)
                all_features.append(extract_mfcc_2d(y_pitched, sr))
                all_labels.append(label)

                # 5. Time Stretch (Ralenti 0.8)
                y_slow = time_stretch(y_proc, rate=0.8)
                all_features.append(extract_mfcc_2d(y_slow, sr))
                all_labels.append(label)

                # 6. Time Stretch (Accéléré 1.2)
                y_fast = time_stretch(y_proc, rate=1.2)
                all_features.append(extract_mfcc_2d(y_fast, sr))
                all_labels.append(label)

        except Exception as e:
            print(f"\n⚠ Erreur sur {file_path}: {e}")

    # ─── Conversion et sauvegarde ─────────────────────────────────────────
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels,   dtype=np.int32)

    # Ajouter la dimension de canal pour le CNN : (N, 120, 216) → (N, 120, 216, 1)
    X = X[..., np.newaxis]

    print(f"\n{'='*60}")
    print(f"  EXTRACTION TERMINEE")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Distribution des emotions :")
    for i, name in EMOTION_MAP.items():
        count = np.sum(y == (i-1))
        print(f"    [{i-1}] {name:<12} -> {count} echantillons")
    print(f"{'='*60}\n")

    # Sauvegarder les arrays
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    x_filename = f"X_{prefix}cnn.npy"
    y_filename = f"y_{prefix}labels.npy"
    np.save(os.path.join(OUTPUT_PATH, x_filename), X)
    np.save(os.path.join(OUTPUT_PATH, y_filename), y)

    # Sauvegarder le label map
    with open(os.path.join(OUTPUT_PATH, 'label_map.pkl'), 'wb') as f:
        pickle.dump(EMOTION_MAP, f)

    print("OK - Fichiers sauvegardes :")
    print(f"   -> {x_filename}  {X.shape}")
    print(f"   -> {y_filename}  {y.shape}")
    print(f"   -> label_map.pkl")

    # update manifest of used curated files when --new-only
    if args.new_only:
        try:
            curated_root = os.path.join(DATA_PATH, ADMIN_CURATED_DIR)
            used = set(used_set)
            for path, _ in all_files:
                rel = os.path.relpath(path, curated_root)
                used.add(rel)
            with open(curated_manifest_path, 'w', encoding='utf-8') as mf:
                json.dump(sorted(list(used)), mf)
        except Exception as e:
            print(f"Warning: failed to update curated manifest: {e}")


if __name__ == "__main__":
    main()
