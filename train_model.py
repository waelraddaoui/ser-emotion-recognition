import os
import sys
import pickle
import numpy as np
print("Script started...")

# Configuration environnement
os.environ["PYTHONUTF8"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Note: on laisse TensorFlow choisir automatiquement GPU ou CPU

# Import TensorFlow BEFORE sklearn (critical on Windows to avoid crashes)
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configuration
BASE_PATH    = os.getcwd()
MODELS_DIR   = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_CLASSES  = 8
BATCH_SIZE = 32   # Batch plus grand pour une convergence plus stable
EPOCHS     = 150  # Epochs suffisants avec early stopping
SEED       = 42

def build_deep_cnn(input_shape, n_classes):
    """
    Architecture CNN optimisée pour SER (Speech Emotion Recognition).
    Corrections : dropout réduit en debut, régularisation L2 ajoutée,
    LR initial plus faible pour éviter le collapse de classe.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Bloc 1 — Capture des patterns locaux
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.20),

        # Bloc 2 — Patterns de niveau intermédiaire
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc 3 — Features émotionnelles complexes
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # Bloc 4 — Représentation globale
        layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        # Classification
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.40),
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.30),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def main():
    print("--- Starting Advanced CNN Training ---")

    # Load Data
    print("Loading data...")
    X = np.load('X_cnn.npy').astype('float32')
    y = np.load('y_labels.npy').astype('int32')

    # ── Sécurité : filtrer les labels hors-plage (> 5) ──────────────────────
    valid_mask = (y >= 0) & (y < N_CLASSES)
    if not np.all(valid_mask):
        print(f"  ⚠ {np.sum(~valid_mask)} échantillons avec labels invalides supprimés.")
        X = X[valid_mask]
        y = y[valid_mask]
    print(f"  Dataset : {X.shape[0]} échantillons, {N_CLASSES} classes")
    print(f"  Distribution : { {i: int(np.sum(y==i)) for i in range(N_CLASSES)} }")

    # Split stratifié
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalisation par la moyenne et l'écart-type du set d'entraînement
    m = np.mean(X_train, axis=0, keepdims=True)
    s = np.std(X_train, axis=0, keepdims=True) + 1e-8
    X_train = (X_train - m) / s
    X_val   = (X_val   - m) / s
    X_test  = (X_test  - m) / s

    # Sauvegarde des stats de normalisation et du set de test
    np.save(os.path.join(MODELS_DIR, 'norm_mean.npy'), m)
    np.save(os.path.join(MODELS_DIR, 'norm_std.npy'),  s)
    np.save(os.path.join(MODELS_DIR, 'X_test.npy'),    X_test)
    np.save(os.path.join(MODELS_DIR, 'y_test.npy'),    y_test)
    
    # Compute class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))
    
    # Build & Compile
    print("Building model...")
    model = build_deep_cnn(X_train.shape[1:], N_CLASSES)
    # LR initial à 3e-4 (plus stable que 1e-3 pour SER)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'best_model.keras'),
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8,
                          min_lr=1e-7, verbose=1)
    ]
    
    # Train
    print("Beginning fit (Deep CNN)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save History
    with open(os.path.join(MODELS_DIR, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)
    
    print("CNN Training Complete.")

if __name__ == "__main__":
    main()
