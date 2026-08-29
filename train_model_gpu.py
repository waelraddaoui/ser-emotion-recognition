"""
=============================================================================
train_model_gpu.py — Entraînement CNN accéléré GPU (PyTorch)
=============================================================================
Titre du PFE : "Classification des Émotions Vocales à partir de MFCCs
               à l'aide de Réseaux Neuronaux Convolutifs :
               Une Étude sur le Corpus RAVDESS"

Version PyTorch du CNN pour exploiter le GPU (RTX 2050).
Identique au train_model.py en termes d'architecture, mais 10-20x plus
rapide grâce à l'accélération GPU via CUDA.

Sorties :
  - models/best_model_gpu.pth   : poids PyTorch
  - models/norm_mean.npy        : moyenne de normalisation
  - models/norm_std.npy         : écart-type de normalisation
  - models/X_test.npy           : set de test
  - models/y_test.npy           : labels de test
  - models/training_history_gpu.pkl : historique d'entraînement
=============================================================================
"""

import os
import sys
import argparse

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Disable output buffering for live progress on Windows
os.environ['PYTHONUNBUFFERED'] = '1'
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_CLASSES  = 8
BATCH_SIZE = 64
EPOCHS     = 100    # OneCycleLR est souvent plus rapide
SEED       = 42
PATIENCE   = 15

torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PyTorch emotion CNN")
    parser.add_argument("--features", default="X_cnn.npy", help="Path to CNN feature array")
    parser.add_argument("--labels", default="y_labels.npy", help="Path to label array")
    parser.add_argument("--model-output", default=os.path.join(MODELS_DIR, "best_model_gpu.pth"), help="Where to save the best model")
    parser.add_argument("--initial-model", default="", help="Optional existing model weights to fine-tune from")
    return parser.parse_args()


def split_dataset(X, y):
    unique, counts = np.unique(y, return_counts=True)
    if len(y) < 10 or len(unique) < 2 or counts.min() < 2:
        print("[WARN] Small or single-class admin dataset detected; using all samples for train/val/test.")
        return X, X, X, y, y, y

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=SEED,
            stratify=y,
        )
        temp_unique, temp_counts = np.unique(y_temp, return_counts=True)
        stratify_temp = y_temp if len(temp_unique) > 1 and temp_counts.min() >= 2 else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=SEED,
            stratify=stratify_temp,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    except ValueError as exc:
        print(f"[WARN] Could not create stratified splits ({exc}); using all samples for train/val/test.")
        return X, X, X, y, y, y


def build_class_weights(y_train):
    present_classes = np.unique(y_train)
    weights = np.ones(N_CLASSES, dtype=np.float32)
    if len(present_classes) > 1:
        computed = compute_class_weight("balanced", classes=present_classes, y=y_train)
        for class_id, weight in zip(present_classes, computed):
            if 0 <= class_id < N_CLASSES:
                weights[class_id] = weight
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE CNN (équivalente à train_model.py, format PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Bloc robuste : Conv2d -> BN -> LeakyReLU -> Conv2d -> BN -> LeakyReLU -> MaxPool -> Dropout"""
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            # Connection résiduelle simple simulée par la structure
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class EmotionCNN(nn.Module):
    """CNN Progressif 5 blocs pour viser 80%."""

    def __init__(self, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32,   dropout=0.15),   # (B, 32,  60, 108)
            ConvBlock(32, 64,  dropout=0.20),   # (B, 64,  30, 54)
            ConvBlock(64, 128, dropout=0.25),   # (B, 128, 15, 27)
            ConvBlock(128, 256, dropout=0.30),  # (B, 256, 7, 13)
            # Bloc 5
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        correct    += (outputs.argmax(1) == y_batch).sum().item()
        total      += X_batch.size(0)
    return total_loss / total, correct / total

def train_epoch_with_scheduler(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * X_batch.size(0)
        correct    += (outputs.argmax(1) == y_batch).sum().item()
        total      += X_batch.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            correct    += (outputs.argmax(1) == y_batch).sum().item()
            total      += X_batch.size(0)
    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Entraînement sur : {device}")

    # 1. Charger les données
    print("[1] Chargement des données...")
    try:
        X = np.load(args.features)
        y = np.load(args.labels)
    except FileNotFoundError:
        print("Erreur : Fichiers .npy non trouvés. Lancez prepare_data.py d'abord.")
        return

    if len(X) == 0 or len(y) == 0:
        print("Erreur : dataset vide, entrainement impossible.")
        return

    print(f"  Features file : {args.features}")
    print(f"  Labels file   : {args.labels}")
    print(f"  Samples       : {len(y)}")
    print(f"  Classes       : {sorted(np.unique(y).tolist())}")

    # Normalisation globale pour stabiliser les gradients
    X_mean = X.mean()
    X_std  = X.std()
    X = (X - X_mean) / (X_std + 1e-6)
    
    # Split stratifie si possible; report-only batches can be tiny.
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # Conversion Tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train).permute(0, 3, 1, 2), torch.LongTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val).permute(0, 3, 1, 2),   torch.LongTensor(y_val))
    test_ds  = TensorDataset(torch.FloatTensor(X_test).permute(0, 3, 1, 2),  torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # 2. Modèle et Optimisation
    model = EmotionCNN(n_classes=N_CLASSES).to(device)
    if args.initial_model:
        if os.path.exists(args.initial_model):
            model.load_state_dict(torch.load(args.initial_model, map_location=device, weights_only=True))
            print(f"[INFO] Fine-tuning from existing model: {args.initial_model}")
        else:
            print(f"[WARN] Initial model not found, training from scratch: {args.initial_model}")
    
    # Pondération des classes
    class_weights = build_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Scheduler OneCycleLR (très puissant pour atteindre les sommets de précision)
    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=total_steps)

    # 3. Boucle d'entraînement
    print(f"[2] Début de l'entraînement ({EPOCHS} époques max, OneCycleLR active)...")
    best_val_acc = 0.0
    no_improve = 0

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    best_path = args.model_output
    os.makedirs(os.path.dirname(os.path.abspath(best_path)), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # On passe le scheduler à train_epoch pour l'update par batch
        tr_loss, tr_acc = train_epoch_with_scheduler(model, train_loader, criterion, optimizer, scheduler, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        history['loss'].append(tr_loss)
        history['accuracy'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_accuracy'].append(vl_acc)

        print(f"Epoch {epoch:3d}/{EPOCHS} -- "
              f"loss: {tr_loss:.4f}, acc: {tr_acc:.4f}  |  "
              f"val_loss: {vl_loss:.4f}, val_acc: {vl_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        # ModelCheckpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save(model.state_dict(), best_path)
            print(f"  [OK] Meilleur modele sauvegarde (val_acc={vl_acc:.4f})", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[STOP] Early stopping declenche a l'epoch {epoch} (patience={PATIENCE})")
                break

    # ── Évaluation finale ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ÉVALUATION FINALE (meilleur modèle)")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(best_path, weights_only=True))
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"  Test Accuracy : {te_acc*100:.2f}%")
    print(f"  Test Loss     : {te_loss:.4f}")

    # ── Sauvegardes ───────────────────────────────────────────────────────
    # Historique d'entraînement
    with open(os.path.join(MODELS_DIR, 'training_history_gpu.pkl'), 'wb') as f:
        pickle.dump(history, f)


    # Statistiques de normalisation (INDISPENSABLE pour l'inférence)
    np.save(os.path.join(MODELS_DIR, 'norm_mean.npy'), X_mean)
    np.save(os.path.join(MODELS_DIR, 'norm_std.npy'), X_std)
    print(f"  Stats de normalisation sauvegardées dans models/")

    # Export ONNX pour compatibilité future si besoin
    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT TERMINÉ")
    print(f"  Meilleure val_accuracy : {best_val_acc*100:.2f}%")
    print(f"  Test accuracy          : {te_acc*100:.2f}%")
    print(f"  Modèle sauvegardé dans : {best_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
