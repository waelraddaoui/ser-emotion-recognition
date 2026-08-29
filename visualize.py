"""
=============================================================================
visualize.py — Visualisations pour le Rapport de Mémoire (RAVDESS SER)
=============================================================================
Génère des figures publication-ready pour le mémoire de master :
  1. Distribution des émotions dans le dataset
  2. Exemples de MFCCs par émotion (grille 8×1)
  3. Comparaison MFCC / Delta / Delta-Delta
  4. Waveforms et spectrogrammes par émotion
=============================================================================
"""

import os
import pickle
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_PATH   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_PATH, "data")
FIGURES_DIR = os.path.join(BASE_PATH, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

SAMPLE_RATE = 22050
DURATION    = 2.5
OFFSET      = 0.5

EMOTION_NAMES = [
    "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
]

EMOTION_EMOJIS = ["😐", "😌", "😊", "😢", "😠", "😨", "🤢", "😲"]

COLORS = [
    '#607D8B', '#9C27B0', '#FFC107', '#2196F3',
    '#F44336', '#673AB7', '#4CAF50', '#FF6F00'
]

# Correspondance émotion code RAVDESS (3ème segment) -> index
EMOTION_CODE_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def find_example_file(emotion_code):
    """Trouve un fichier WAV exemple pour une émotion donnée."""
    for actor_dir in sorted(os.listdir(DATA_PATH)):
        actor_path = os.path.join(DATA_PATH, actor_dir)
        if not os.path.isdir(actor_path) or not actor_dir.startswith("Actor_"):
            continue
        for fname in sorted(os.listdir(actor_path)):
            if fname.endswith('.wav'):
                parts = fname.split('-')
                if len(parts) >= 7 and int(parts[2]) == emotion_code:
                    return os.path.join(actor_path, fname)
    return None


def load_audio(file_path):
    """Charge un fichier audio avec les paramètres standard."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, offset=OFFSET)
    return y, sr


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 : Distribution des émotions
# ─────────────────────────────────────────────────────────────────────────────

def plot_emotion_distribution():
    """Compte et visualise la distribution des émotions dans RAVDESS."""
    counts = {name: 0 for name in EMOTION_NAMES}
    emotion_code_to_name = {k: EMOTION_NAMES[v] for k, v in EMOTION_CODE_MAP.items()}

    for actor_dir in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for fname in os.listdir(actor_path):
            if fname.endswith('.wav'):
                parts = fname.split('-')
                if len(parts) >= 7:
                    code = int(parts[2])
                    if code in emotion_code_to_name:
                        counts[emotion_code_to_name[code]] += 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Distribution des Émotions dans le Corpus RAVDESS",
        fontsize=14, fontweight='bold'
    )

    names  = list(counts.keys())
    values = list(counts.values())

    # Bar chart
    bars = axes[0].bar(names, values, color=COLORS, edgecolor='white', linewidth=1.5, zorder=3)
    axes[0].set_xlabel('Émotion', fontsize=12)
    axes[0].set_ylabel("Nombre d'échantillons", fontsize=12)
    axes[0].set_title('Répartition Absolue', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=35)
    axes[0].grid(axis='y', alpha=0.4, zorder=0)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        values, labels=names, colors=COLORS, autopct='%1.1f%%',
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[1].set_title('Répartition Proportionnelle', fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'emotion_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Distribution des émotions : {path}")
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 : Grille MFCC par émotion
# ─────────────────────────────────────────────────────────────────────────────

def plot_mfcc_grid():
    """Affiche un spectrogramme MFCC pour chaque emotion (grille 2x4)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(
        "Spectrogrammes MFCC par Emotion — Corpus RAVDESS\n"
        "40 Coefficients Mel-Frequency Cepstral (n_mfcc=40)",
        fontsize=13, fontweight='bold'
    )

    for idx, (emotion_name, emoji, color) in enumerate(zip(EMOTION_NAMES, EMOTION_EMOJIS, COLORS)):
        emotion_code = idx + 1
        ax = axes[idx // 4][idx % 4]

        file_path = find_example_file(emotion_code)
        if file_path is None:
            ax.text(0.5, 0.5, "Introuvable", ha='center', va='center')
            ax.set_title(f"{emoji} {emotion_name}")
            continue

        y, sr = load_audio(file_path)
        mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        img = librosa.display.specshow(
            mfcc, sr=sr, x_axis='time', ax=ax,
            cmap='magma'
        )
        ax.set_title(f"{emoji} {emotion_name}", fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('Temps (s)', fontsize=9)
        ax.set_ylabel('Coefficients MFCC', fontsize=9)
        plt.colorbar(img, ax=ax, format='%+2.0f')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'mfcc_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Grille MFCC par émotion : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 : MFCC vs Delta vs Delta-Delta (une émotion exemple)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mfcc_delta_comparison():
    """Compare les 3 types de features MFCC sur un exemple."""
    # Prendre un fichier "Joyeux" comme exemple
    file_path = find_example_file(3)  # Happy = code 3
    if file_path is None:
        print("  ⚠ Fichier exemple introuvable pour la comparaison MFCC/Delta.")
        return

    y, sr = load_audio(file_path)
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2= librosa.feature.delta(mfcc, order=2)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        "Comparaison MFCC / Δ MFCC / ΔΔ MFCC\n"
        "Émotion : Happy 😊 (Acteur RAVDESS)",
        fontsize=13, fontweight='bold'
    )

    titles = ["MFCC (Base)", "Δ MFCC (Delta — Dynamique temporelle)", "ΔΔ MFCC (Delta-Delta — Accélération)"]
    features = [mfcc, mfcc_delta, mfcc_delta2]
    cmaps    = ['Blues', 'Greens', 'Oranges']

    for ax, feat, title, cmap in zip(axes, features, titles, cmaps):
        img = librosa.display.specshow(feat, sr=sr, x_axis='time', ax=ax, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Coefficients', fontsize=10)
        plt.colorbar(img, ax=ax)

    axes[-1].set_xlabel('Temps (s)', fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'mfcc_delta_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Comparaison MFCC/Delta/Delta² : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 : Waveforms comparées par émotion
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform_grid():
    """Affiche les formes d'onde de 8 emotions cote a cote."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 7))
    fig.suptitle(
        "Formes d'Onde Audio par Emotion — Corpus RAVDESS",
        fontsize=13, fontweight='bold'
    )

    for idx, (emotion_name, emoji, color) in enumerate(zip(EMOTION_NAMES, EMOTION_EMOJIS, COLORS)):
        emotion_code = idx + 1
        ax = axes[idx // 4][idx % 4]

        file_path = find_example_file(emotion_code)
        if file_path is None:
            ax.text(0.5, 0.5, "Introuvable", ha='center', va='center')
            continue

        y, sr = load_audio(file_path)
        times = np.linspace(0, len(y) / sr, num=len(y))

        ax.plot(times, y, color=color, linewidth=0.8, alpha=0.9)
        ax.fill_between(times, y, alpha=0.2, color=color)
        ax.set_title(f"{emoji} {emotion_name}", fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('Temps (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'waveform_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Grille de waveforms : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 : Architecture CNN (schéma textuel visuel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cnn_architecture():
    """Génère un schéma visuel de l'architecture CNN."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    fig.suptitle(
        "Architecture du Modèle CNN — Classification d'Émotions Vocales (RAVDESS)",
        fontsize=12, fontweight='bold', color='white', y=0.98
    )

    layers_info = [
        ("Input\n(120x216x1)", '#37474F',   0.5),
        ("Conv2D 32\nBN + ReLU\nMaxPool\nDropout 0.25", '#1565C0', 2.5),
        ("Conv2D 64\nBN + ReLU\nMaxPool\nDropout 0.25", '#1976D2', 4.5),
        ("Conv2D 128\nBN + ReLU\nMaxPool\nDropout 0.30", '#1E88E5', 6.5),
        ("Conv2D 256\nBN + ReLU\nGAP",       '#2196F3', 8.5),
        ("Dense 256\nBN + ReLU\nDropout 0.5", '#43A047', 10.5),
        ("Dense 128\nReLU\nDropout 0.3",      '#66BB6A', 12.5),
        ("Output\n6 classes\nSoftmax",         '#E53935', 14.5),
    ]

    for label, color, x in layers_info:
        rect = plt.Rectangle((x - 0.8, 1.0), 1.6, 3.0,
                              color=color, alpha=0.9, zorder=3, linewidth=2,
                              edgecolor='white')
        ax.add_patch(rect)
        ax.text(x, 2.5, label, ha='center', va='center',
                fontsize=8, color='white', fontweight='bold', zorder=4)

    # Flèches
    for _, _, x in layers_info[:-1]:
        ax.annotate('', xy=(x + 0.85, 2.5), xytext=(x + 0.7, 2.5),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cnn_architecture.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  → Schéma architecture CNN : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  PFE — Génération des Visualisations (RAVDESS SER)")
    print(f"  Figures sauvegardées dans : {FIGURES_DIR}")
    print(f"{'='*60}\n")

    print("  [1/5] Distribution des émotions...")
    plot_emotion_distribution()

    print("  [2/5] Grille MFCC par émotion...")
    plot_mfcc_grid()

    print("  [3/5] Comparaison MFCC / Delta / Delta²...")
    plot_mfcc_delta_comparison()

    print("  [4/5] Grille de waveforms...")
    plot_waveform_grid()

    print("  [5/5] Architecture CNN...")
    plot_cnn_architecture()

    print(f"\n{'='*60}")
    print("  VISUALISATIONS TERMINÉES")
    print(f"  5 figures dans : {FIGURES_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
