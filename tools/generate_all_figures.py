"""
Generate all figures and diagrams for the PFE report.
"""
import os
import sys
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(BASE, "figures")
MODELS = os.path.join(BASE, "models")
sys.path.insert(0, os.path.join(BASE, "backend"))

os.makedirs(FIGURES, exist_ok=True)

EMOTIONS = [
    "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
]
COLORS = [
    "#607D8B", "#9C27B0", "#FFC107", "#2196F3",
    "#F44336", "#673AB7", "#4CAF50", "#FF6F00",
]


def add_box(ax, x, y, w, h, text, facecolor, edgecolor, textcolor="#0f172a", fontsize=10, lw=1.6):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, fontweight="bold", color=textcolor,
    )
    return box


def add_arrow(ax, start, end, color="#334155", lw=1.6, style="->", curved=False):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12, linewidth=lw,
        color=color, connectionstyle="arc3,rad=0.12" if curved else "arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    return arrow


def plot_pipeline_architecture():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fc")

    boxes = [
        (0.3, 2.2, "Audio\n(WAV/MP3)"),
        (2.0, 2.2, "Prétraitement\n22.05 kHz, trim"),
        (3.7, 2.2, "MFCC +\nΔ + Δ²"),
        (5.4, 2.2, "CNN\nEmotionCNN"),
        (7.1, 2.2, "Softmax\n8 classes"),
        (8.8, 2.2, "API\nFastAPI"),
        (10.3, 2.2, "UI\nAngular"),
    ]
    for x, y, text in boxes:
        box = FancyBboxPatch(
            (x, y), 1.4, 1.2, boxstyle="round,pad=0.08",
            facecolor="#e3f2fd", edgecolor="#1565c0", linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + 0.7, y + 0.6, text, ha="center", va="center", fontsize=9, fontweight="bold")

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.4
        x2 = boxes[i + 1][0]
        ax.annotate(
            "", xy=(x2, 2.8), xytext=(x1, 2.8),
            arrowprops=dict(arrowstyle="->", color="#37474f", lw=2),
        )

    ax.set_title(
        "Architecture globale du système SER",
        fontsize=14, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    path = os.path.join(FIGURES, "pipeline_architecture.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> {path}")


def plot_cnn_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("#0f0f23")

    layers = [
        ("Input\n(1×120×216)", "#00BCD4"),
        ("ConvBlock×4\n32→512", "#7C4DFF"),
        ("GAP", "#FF6F00"),
        ("FC 512→256", "#FFC107"),
        ("Dropout", "#9E9E9E"),
        ("Output\n8 classes", "#4CAF50"),
    ]
    x = 0.4
    for label, color in layers:
        box = FancyBboxPatch(
            (x, 1.2), 1.5, 1.6, boxstyle="round,pad=0.06",
            facecolor=color, edgecolor="white", alpha=0.85, linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(x + 0.75, 2.0, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        x += 1.75

    ax.set_title("Architecture EmotionCNN (PyTorch)", color="white", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES, "cnn_architecture.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> {path}")


def plot_ser_system_overview():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    center = FancyBboxPatch(
        (3.2, 2.5), 3.6, 1.4, boxstyle="round,pad=0.1",
        facecolor="#1565c0", edgecolor="none",
    )
    ax.add_patch(center)
    ax.text(5, 3.2, "SER Engine\n(MFCC + CNN)", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")

    satellites = [
        (0.5, 4.5, "Corpus\nRAVDESS"),
        (0.5, 1.0, "Entraînement\nGPU PyTorch"),
        (7.5, 4.5, "Utilisateur\nUpload audio"),
        (7.5, 1.0, "Historique\n& Admin"),
    ]
    for x, y, t in satellites:
        b = FancyBboxPatch(
            (x, y), 2.2, 1.0, boxstyle="round,pad=0.05",
            facecolor="#e8eaf6", edgecolor="#3949ab", linewidth=1.5,
        )
        ax.add_patch(b)
        ax.text(x + 1.1, y + 0.5, t, ha="center", va="center", fontsize=9)

    ax.set_title("Figure 1 — Système de reconnaissance des émotions vocales", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES, "ser_system_overview.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_use_case_diagram():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    system = FancyBboxPatch(
        (2.3, 0.9), 8.4, 5.2, boxstyle="round,pad=0.12,rounding_size=0.08",
        facecolor="#eef2ff", edgecolor="#4f46e5", linewidth=2.2,
    )
    ax.add_patch(system)
    ax.text(6.5, 6.0, "Système SER Studio", ha="center", va="center", fontsize=13, fontweight="bold", color="#1e3a8a")

    add_box(ax, 0.5, 4.8, 1.3, 0.7, "Utilisateur", "#dbeafe", "#2563eb")
    add_box(ax, 0.5, 1.6, 1.3, 0.7, "Administrateur", "#ede9fe", "#7c3aed")
    add_box(ax, 11.2, 4.8, 1.3, 0.7, "Backend", "#dcfce7", "#16a34a")

    use_cases = [
        (3.0, 4.8, 2.0, 0.8, "Se connecter"),
        (5.5, 4.8, 2.2, 0.8, "Analyser un audio"),
        (8.1, 4.8, 2.2, 0.8, "Consulter l'historique"),
        (3.0, 2.2, 2.0, 0.8, "Signaler une erreur"),
        (5.5, 2.2, 2.2, 0.8, "Écouter l'audio"),
        (8.1, 2.2, 2.2, 0.8, "Vérifier / retester"),
    ]
    for x, y, w, h, label in use_cases:
        ellipse = mpatches.Ellipse(
            (x + w / 2, y + h / 2), width=w, height=h,
            facecolor="#ffffff", edgecolor="#475569", linewidth=1.6,
        )
        ax.add_patch(ellipse)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color="#0f172a")

    # Connections from actors to use cases
    add_arrow(ax, (1.8, 5.15), (3.0, 5.15))
    add_arrow(ax, (1.8, 4.95), (5.5, 5.15))
    add_arrow(ax, (1.8, 1.95), (3.0, 2.6))
    add_arrow(ax, (11.2, 5.15), (10.3, 5.15))
    add_arrow(ax, (11.2, 4.95), (10.3, 2.6))

    ax.text(2.55, 5.75, "Fonctions utilisateur", fontsize=10, fontweight="bold", color="#334155")
    ax.text(7.95, 5.75, "Fonctions d'analyse", fontsize=10, fontweight="bold", color="#334155")
    ax.text(2.55, 1.25, "Signalement", fontsize=10, fontweight="bold", color="#334155")
    ax.text(7.95, 1.25, "Fonctions admin", fontsize=10, fontweight="bold", color="#334155")

    ax.set_title("Diagramme de cas d'utilisation du système SER", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES, "use_case_diagram.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> {path}")


def plot_sequence_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    participants = [
        (1.2, "Utilisateur", "#dbeafe", "#2563eb"),
        (3.7, "Frontend Angular", "#e9d5ff", "#7c3aed"),
        (6.2, "FastAPI", "#dcfce7", "#16a34a"),
        (8.7, "Service ML", "#fef3c7", "#d97706"),
        (11.2, "SQLite / Stockage", "#fee2e2", "#dc2626"),
    ]
    y_top = 5.7
    y_bottom = 0.8

    for x, label, face, edge in participants:
        add_box(ax, x - 0.95, 6.1, 1.9, 0.65, label, face, edge, fontsize=9.5)
        ax.plot([x, x], [y_top, y_bottom], linestyle="--", color="#94a3b8", linewidth=1.2)

    def msg(y, x1, x2, label, color="#334155"):
        ax.annotate(
            "", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8),
        )
        ax.text((x1 + x2) / 2, y + 0.18, label, ha="center", va="bottom", fontsize=9, color="#0f172a")

    msg(5.0, 1.2, 3.7, "1. Upload audio")
    msg(4.45, 3.7, 6.2, "2. POST /predict")
    msg(3.9, 6.2, 8.7, "3. Prétraitement + inférence")
    msg(3.25, 8.7, 6.2, "4. Classe + confiance")
    msg(2.6, 6.2, 11.2, "5. Sauvegarde de la prédiction")
    msg(2.0, 6.2, 3.7, "6. Résultat JSON")
    msg(1.45, 3.7, 1.2, "7. Affichage final")

    ax.text(6.5, 0.25, "Scénario principal : de l'upload jusqu'à l'affichage du résultat", ha="center", va="center", fontsize=10, color="#475569")
    ax.set_title("Diagramme de séquence — prédiction d'une émotion vocale", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES, "sequence_diagram.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {path}")


def plot_deployment_diagram():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    add_box(ax, 0.4, 4.4, 2.2, 1.1, "Client Web\n(Navigateur / Mobile)", "#dbeafe", "#2563eb", fontsize=9.5)
    add_box(ax, 3.2, 4.4, 2.3, 1.1, "Angular SPA\nPort 4200", "#ede9fe", "#7c3aed", fontsize=9.5)
    add_box(ax, 6.1, 4.4, 2.4, 1.1, "FastAPI Backend\nPort 8001", "#dcfce7", "#16a34a", fontsize=9.5)
    add_box(ax, 9.1, 4.4, 2.4, 1.1, "Service ML\nEmotionCNN / PyTorch", "#fef3c7", "#d97706", fontsize=9.5)

    add_box(ax, 3.8, 1.6, 2.2, 1.0, "SQLite\napp.db", "#fee2e2", "#dc2626", fontsize=9.5)
    add_box(ax, 6.7, 1.6, 2.2, 1.0, "Dossier uploads\nAudio stockés", "#e0f2fe", "#0284c7", fontsize=9.5)
    add_box(ax, 9.6, 1.6, 2.2, 1.0, "Modèle entraîné\n.pth / .npy", "#ede9fe", "#6d28d9", fontsize=9.5)

    add_arrow(ax, (2.6, 5.0), (3.2, 5.0))
    add_arrow(ax, (5.5, 5.0), (6.1, 5.0))
    add_arrow(ax, (8.5, 5.0), (9.1, 5.0))
    add_arrow(ax, (7.3, 4.4), (5.0, 2.6), curved=True)
    add_arrow(ax, (8.0, 4.4), (7.8, 2.6), curved=True)
    add_arrow(ax, (8.5, 5.0), (10.1, 2.6), curved=True)

    ax.text(6.5, 6.25, "Architecture de déploiement de la plateforme SER", ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax.text(6.5, 0.7, "Le backend orchestre l'inférence, la persistance des prédictions et la gestion des signalements.", ha="center", va="center", fontsize=10, color="#475569")
    path = os.path.join(FIGURES, "deployment_diagram.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> {path}")


def plot_class_diagram():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    def class_box(x, y, w, h, title, attrs, color):
        outer = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
                               facecolor="#ffffff", edgecolor=color, linewidth=1.8)
        ax.add_patch(outer)
        header = FancyBboxPatch((x, y + h - 0.55), w, 0.55, boxstyle="round,pad=0.04,rounding_size=0.06",
                                facecolor=color, edgecolor=color, linewidth=1.0)
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        if attrs:
            ax.text(x + 0.14, y + h - 0.68, "\n".join(attrs), ha="left", va="top", fontsize=8.5, color="#0f172a")

    class_box(0.6, 4.4, 2.2, 1.7, "User", ["id", "email", "role", "is_active"], "#2563eb")
    class_box(3.3, 4.4, 2.4, 1.7, "Prediction", ["id", "emotion", "confidence", "audio_id", "user_id"], "#7c3aed")
    class_box(6.2, 4.4, 2.4, 1.7, "PredictionReport", ["id", "prediction_id", "reason", "status", "reviewed_by"], "#dc2626")
    class_box(9.1, 4.4, 2.2, 1.7, "AudioFile", ["id", "filename", "path", "created_at"], "#16a34a")
    class_box(3.4, 1.5, 2.6, 1.5, "EmotionService", ["preprocess()", "predict()", "save_result()"], "#d97706")

    add_arrow(ax, (2.8, 5.0), (3.3, 5.0))
    add_arrow(ax, (5.7, 5.0), (6.2, 5.0))
    add_arrow(ax, (5.7, 4.75), (9.1, 4.75))
    add_arrow(ax, (8.6, 5.0), (9.1, 5.0))
    add_arrow(ax, (4.7, 4.4), (4.7, 3.0), curved=False)
    add_arrow(ax, (4.7, 3.0), (4.7, 4.4), curved=False)

    ax.text(1.7, 6.55, "Utilisateur", fontsize=10, fontweight="bold", color="#1d4ed8", ha="center")
    ax.text(4.5, 6.55, "Prédiction", fontsize=10, fontweight="bold", color="#6d28d9", ha="center")
    ax.text(7.4, 6.55, "Signalement", fontsize=10, fontweight="bold", color="#b91c1c", ha="center")
    ax.text(10.2, 6.55, "Fichier audio", fontsize=10, fontweight="bold", color="#15803d", ha="center")

    ax.text(6.5, 0.6, "Diagramme de classes simplifié des entités métier et des relations principales", ha="center", va="center", fontsize=10, color="#475569")
    ax.set_title("Diagramme de classes — noyau métier SER", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(FIGURES, "class_diagram.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> {path}")


def evaluate_pytorch_model():
    """Confusion matrix and metrics from PyTorch model on 8-class test split."""
    from sklearn.model_selection import train_test_split

    x_all = np.load(os.path.join(BASE, "X_cnn.npy"))
    y_all = np.load(os.path.join(BASE, "y_labels.npy"))
    _, x_test, _, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42, stratify=y_all,
    )
    print(f"  Evaluating on {len(y_test)} samples (8 classes)")

    import torch
    prev = os.getcwd()
    os.chdir(os.path.join(BASE, "backend"))
    try:
        from app import ml
        ml.load_model()
    finally:
        os.chdir(prev)
    model = ml._model
    mean = ml._norm_mean
    std = ml._norm_std

    preds = []
    batch = 32
    if x_test.ndim == 4 and x_test.shape[-1] == 1:
        x_test = np.transpose(x_test, (0, 3, 1, 2))
    elif x_test.ndim == 3:
        x_test = x_test[:, np.newaxis, :, :]
    print(f"  X_test shape: {x_test.shape}")

    for i in range(0, len(x_test), batch):
        chunk = x_test[i : i + batch].astype(np.float32)
        if mean is not None and std is not None:
            if np.size(mean) == 1:
                chunk = (chunk - float(mean)) / (float(std) + 1e-6)
            else:
                chunk = (chunk - float(np.mean(mean))) / (float(np.mean(std)) + 1e-6)
        t = torch.from_numpy(chunk)
        with torch.no_grad():
            out = model(t)
            preds.extend(torch.argmax(out, dim=1).numpy().tolist())

    y_pred = np.array(preds)
    y_true = y_test.astype(int)
    acc = (y_pred == y_true).mean()
    print(f"  Test accuracy (PyTorch): {acc * 100:.2f}%")

    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax,
    )
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Vérité terrain")
    ax.set_title(f"Matrice de confusion — Accuracy {acc*100:.1f}%")
    plt.tight_layout()
    p1 = os.path.join(FIGURES, "confusion_matrix.png")
    plt.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {p1}")

    report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True)
    labels = EMOTIONS
    f1s = [report[l]["f1-score"] for l in labels]
    precs = [report[l]["precision"] for l in labels]
    recs = [report[l]["recall"] for l in labels]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precs, w, label="Précision", color="#2196F3")
    ax.bar(x, recs, w, label="Rappel", color="#4CAF50")
    ax.bar(x + w, f1s, w, label="F1-score", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Métriques par classe émotionnelle")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(FIGURES, "per_class_metrics.png")
    plt.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  -> {p2}")

    hist_path = os.path.join(MODELS, "training_history_gpu.pkl")
    if os.path.exists(hist_path):
        with open(hist_path, "rb") as f:
            hist = pickle.load(f)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        epochs = range(1, len(hist.get("train_loss", [])) + 1)
        if "train_loss" in hist:
            ax1.plot(epochs, hist["train_loss"], label="Train")
            ax1.plot(epochs, hist["val_loss"], label="Val")
            ax1.set_title("Loss")
            ax1.legend()
            ax1.grid(alpha=0.3)
        if "train_acc" in hist:
            ax2.plot(epochs, [a * 100 for a in hist["train_acc"]], label="Train")
            ax2.plot(epochs, [a * 100 for a in hist["val_acc"]], label="Val")
            ax2.set_title("Accuracy (%)")
            ax2.legend()
            ax2.grid(alpha=0.3)
        plt.tight_layout()
        p3 = os.path.join(FIGURES, "learning_curves.png")
        plt.savefig(p3, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"  -> {p3}")

    with open(os.path.join(FIGURES, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%\n\n")
        f.write(classification_report(y_true, y_pred, target_names=EMOTIONS))


def run_visualize_script():
    viz = os.path.join(BASE, "visualize.py")
    if os.path.exists(viz):
        import subprocess
        r = subprocess.run(
            [sys.executable, viz],
            cwd=BASE, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            print("  visualize.py OK")
        else:
            print("  visualize.py:", (r.stderr or r.stdout)[:300])


def main():
    print("Generating report figures...")
    plot_ser_system_overview()
    plot_pipeline_architecture()
    plot_use_case_diagram()
    plot_sequence_diagram()
    plot_deployment_diagram()
    plot_class_diagram()
    plot_cnn_architecture()
    run_visualize_script()
    try:
        evaluate_pytorch_model()
    except Exception as e:
        print(f"  PyTorch eval warning: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
