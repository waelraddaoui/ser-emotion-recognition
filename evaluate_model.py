"""
=============================================================================
evaluate_model.py — Évaluation Complète du Modèle CNN (RAVDESS SER)
=============================================================================
Génère toutes les métriques et visualisations pour le rapport de mémoire :
  - Accuracy, Precision, Recall, F1-Score (par classe et global)
  - Matrice de confusion normalisée
  - Courbes ROC multi-classes (One-vs-Rest)
  - Rapport de classification complet
=============================================================================
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_PATH, "models")
RESULTS_DIR = os.path.join(BASE_PATH, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

EMOTION_NAMES = [
    "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
]

N_CLASSES = len(EMOTION_NAMES)  # 8

# Palette couleurs pour les figures
COLORS = [
    '#E91E63', '#9C27B0', '#3F51B5', '#2196F3',
    '#009688', '#8BC34A', '#FF9800', '#795548'
]


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_data():
    """Charge le modèle entraîné et les données de test."""
    model_path = os.path.join(MODELS_DIR, 'best_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Modèle introuvable. Lancez train_model.py d'abord !")

    print("  Chargement du modèle...")
    model = tf.keras.models.load_model(model_path)

    print("  Chargement des données de test...")
    X_test = np.load(os.path.join(MODELS_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(MODELS_DIR, 'y_test.npy'))

    print(f"  Test set : {X_test.shape[0]} échantillons")
    return model, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# MATRICE DE CONFUSION
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    Génère une matrice de confusion normalisée publication-ready.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "Matrice de Confusion — CNN SER (RAVDESS)\n"
        "«Classification des Émotions Vocales à partir de MFCCs»",
        fontsize=13, fontweight='bold', y=1.02
    )

    # Matrice absolue
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
                ax=axes[0], linewidths=0.5, linecolor='gray')
    axes[0].set_title('Nombres Absolus', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Émotion Réelle', fontsize=11)
    axes[0].set_xlabel('Émotion Prédite', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    # Matrice normalisée
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
                ax=axes[1], linewidths=0.5, linecolor='gray',
                vmin=0, vmax=1)
    axes[1].set_title('Proportions Normalisées', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Émotion Réelle', fontsize=11)
    axes[1].set_xlabel('Émotion Prédite', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Matrice de confusion sauvegardée : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# COURBES ROC MULTI-CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_prob, save_dir):
    """
    Trace les courbes ROC pour chaque émotion (One-vs-Rest).
    """
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        "Courbes ROC — CNN SER (RAVDESS)\nOne-vs-Rest par Émotion",
        fontsize=13, fontweight='bold'
    )

    # Courbe macro-average
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(N_CLASSES)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= N_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)

    # Courbes par émotion
    for i, (emotion, color) in enumerate(zip(EMOTION_NAMES, COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.5, alpha=0.7,
                label=f"{emotion} (AUC = {roc_auc:.3f})")

    # Macro-average
    ax.plot(all_fpr, mean_tpr, color='black', linewidth=2.5, linestyle='-',
            label=f"Macro-Average (AUC = {macro_auc:.3f})")

    # Ligne diagonale
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Courbes ROC sauvegardées : {path}")
    return macro_auc


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES PAR CLASSE (BAR CHART)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(y_true, y_pred, save_dir):
    """
    Barre de métriques par émotion : Precision, Recall, F1-Score.
    """
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0
    )

    x = np.arange(N_CLASSES)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Précision', color='#2196F3', alpha=0.85)
    ax.bar(x,         recall,    width, label='Rappel',    color='#4CAF50', alpha=0.85)
    ax.bar(x + width, f1,        width, label='F1-Score',  color='#FF9800', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_NAMES, rotation=30, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        "Métriques par Émotion — CNN SER (RAVDESS)",
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Valeurs au-dessus des barres
    for i in range(N_CLASSES):
        ax.text(i - width, precision[i] + 0.02, f'{precision[i]:.2f}',
                ha='center', va='bottom', fontsize=7, color='#1565C0')
        ax.text(i,         recall[i]    + 0.02, f'{recall[i]:.2f}',
                ha='center', va='bottom', fontsize=7, color='#2E7D32')
        ax.text(i + width, f1[i]        + 0.02, f'{f1[i]:.2f}',
                ha='center', va='bottom', fontsize=7, color='#E65100')

    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Métriques par classe sauvegardées : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# COURBES D'APPRENTISSAGE
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(save_dir):
    """Recharge et trace les courbes d'apprentissage depuis history.pkl."""
    hist_path = os.path.join(MODELS_DIR, 'training_history.pkl')
    if not os.path.exists(hist_path):
        print("  ⚠ Historique d'entraînement introuvable.")
        return

    with open(hist_path, 'rb') as f:
        history = pickle.load(f)

    epochs = range(1, len(history['loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Courbes d'Apprentissage — CNN SER (RAVDESS)",
        fontsize=13, fontweight='bold'
    )

    # Loss
    axes[0].fill_between(epochs, history['loss'],     alpha=0.15, color='#2196F3')
    axes[0].fill_between(epochs, history['val_loss'], alpha=0.15, color='#F44336')
    axes[0].plot(epochs, history['loss'],     label='Train',      color='#2196F3', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation', color='#F44336', linewidth=2, linestyle='--')
    axes[0].set_title('Fonction de Perte (Loss)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Categorical Cross-Entropy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].fill_between(epochs, history['accuracy'],     alpha=0.15, color='#4CAF50')
    axes[1].fill_between(epochs, history['val_accuracy'], alpha=0.15, color='#FF9800')
    axes[1].plot(epochs, history['accuracy'],     label='Train',      color='#4CAF50', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], label='Validation', color='#FF9800', linewidth=2, linestyle='--')
    axes[1].set_title('Précision (Accuracy)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Courbes d'apprentissage sauvegardées : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_report(y_true, y_pred, test_accuracy, macro_auc, save_dir):
    """Sauvegarde un rapport complet en fichier texte."""
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_NAMES,
        digits=4
    )

    report_text = f"""
{'='*65}
  RAPPORT D'ÉVALUATION — CNN SER (RAVDESS)
  Titre : Classification des Émotions Vocales à partir de MFCCs
          à l'aide de Réseaux Neuronaux Convolutifs
{'='*65}

  Test Accuracy     : {test_accuracy*100:.2f}%
  Macro AUC (ROC)   : {macro_auc:.4f}

{'-'*65}
  Rapport de Classification par Émotion
{'-'*65}
{report}
{'='*65}
"""
    path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  → Rapport sauvegardé : {path}")
    print(report_text)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  PFE — Évaluation du Modèle CNN (RAVDESS SER)")
    print(f"{'='*60}\n")

    model, X_test, y_test = load_model_and_data()

    # Prédictions
    print("\n  Calcul des prédictions...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    test_accuracy = np.mean(y_pred == y_test)
    print(f"  Test Accuracy : {test_accuracy*100:.2f}%")

    # Figures
    print("\n  Génération des figures...")
    plot_confusion_matrix(y_test, y_pred, RESULTS_DIR)
    macro_auc = plot_roc_curves(y_test, y_prob, RESULTS_DIR)
    plot_per_class_metrics(y_test, y_pred, RESULTS_DIR)
    plot_learning_curves(RESULTS_DIR)

    # Rapport
    save_metrics_report(y_test, y_pred, test_accuracy, macro_auc, RESULTS_DIR)

    print(f"\n{'='*60}")
    print("  ÉVALUATION TERMINÉE")
    print(f"  Figures sauvegardées dans : {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
