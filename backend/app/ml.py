import io
import os
import base64
import logging
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.io import wavfile
import soundfile as sf
from .config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 22050
DURATION = 2.5
N_MFCC = 40
MAX_FRAMES = 216

EMOTION_NAMES = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
EMOTION_EMOJIS = ["😐", "😌", "😊", "😢", "😠", "😨", "🤢", "😲"]
EMOTION_COLORS = ["#607D8B", "#9C27B0", "#FFC107", "#2196F3", "#F44336", "#673AB7", "#4CAF50", "#FF6F00"]
EMOTION_DESCRIPTIONS = {
    "Neutral": "Neutral tone without strong emotion.",
    "Calm": "Calm and serene expression.",
    "Happy": "Joyful and energetic tone.",
    "Sad": "Slow and melancholic tone.",
    "Angry": "Tense and aggressive tone.",
    "Fearful": "Anxious or trembling tone.",
    "Disgust": "Repulsive or displeased tone.",
    "Surprised": "Sudden and raised intonation.",
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class EmotionCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, dropout=0.15),
            ConvBlock(32, 64, dropout=0.20),
            ConvBlock(64, 128, dropout=0.25),
            ConvBlock(128, 256, dropout=0.30),
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
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


_model = None
_norm_mean = None
_norm_std = None


def load_model():
    global _model, _norm_mean, _norm_std
    if _model is not None:
        return _model

    feedback_model_path = settings.resolve_path(settings.feedback_model_path)
    model_path = feedback_model_path if feedback_model_path.exists() else settings.resolve_path(settings.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    _model = EmotionCNN(n_classes=8)
    state = torch.load(str(model_path), map_location=torch.device("cpu"))
    _model.load_state_dict(state)
    _model.eval()

    mean_path = settings.resolve_path(settings.norm_mean_path)
    std_path = settings.resolve_path(settings.norm_std_path)

    if mean_path.exists() and std_path.exists():
        _norm_mean = np.load(mean_path)
        _norm_std = np.load(std_path)
    return _model


def _normalize(X: np.ndarray) -> np.ndarray:
    if _norm_mean is None or _norm_std is None:
        return X

    mean = _norm_mean
    std = _norm_std
    if np.size(mean) == 1 and np.size(std) == 1:
        return (X - float(mean)) / (float(std) + 1e-6)

    mean_val = float(np.mean(mean))
    std_val = float(np.mean(std))
    return (X - mean_val) / (std_val + 1e-6)


def extract_features(audio_path: str):
    y = None
    sr = SAMPLE_RATE

    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    file_size = os.path.getsize(audio_path)
    logger.debug(f"Processing audio file: {audio_path} (size: {file_size} bytes)")

    try:
        logger.debug("Trying soundfile.read()...")
        data, raw_sr = sf.read(audio_path)
        y = data.astype(np.float32)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        sr = raw_sr
        logger.debug(f"Successfully loaded with soundfile: sr={sr}, shape={y.shape}")
    except Exception as e:
        logger.debug(f"soundfile.read() failed: {type(e).__name__}: {str(e)}")
        try:
            logger.debug("Trying scipy.io.wavfile.read()...")
            raw_sr, data = wavfile.read(audio_path)
            if data.dtype == np.int16:
                y = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                y = data.astype(np.float32) / 2147483648.0
            else:
                y = data.astype(np.float32)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            sr = raw_sr
            logger.debug(f"Successfully loaded with scipy.io.wavfile: sr={sr}, shape={y.shape}")
        except Exception as e2:
            logger.debug(f"scipy.io.wavfile.read() failed: {type(e2).__name__}: {str(e2)}")
            logger.debug("Trying librosa.load()...")
            try:
                y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                logger.debug(f"Successfully loaded with librosa: sr={sr}, shape={y.shape}")
            except Exception as e3:
                logger.error(f"librosa.load() failed: {type(e3).__name__}: {str(e3)}")
                raise RuntimeError(f"All audio loaders failed. Last error: {type(e3).__name__}: {str(e3)}")

    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_norm = librosa.util.normalize(y_trimmed)
    y_filt = librosa.effects.preemphasis(y_norm)

    target_len = int(DURATION * sr)
    if len(y_filt) > target_len:
        y_final = y_filt[:target_len]
    else:
        y_final = librosa.util.pad_center(data=y_filt, size=target_len)

    mfcc = librosa.feature.mfcc(y=y_final, sr=sr, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

    if combined.shape[1] < MAX_FRAMES:
        pad_w = MAX_FRAMES - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_w)), mode="constant")
    else:
        combined = combined[:, :MAX_FRAMES]

    return combined, mfcc, y_final, sr


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_mfcc_spectrogram(mfcc, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="magma")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("MFCC Spectrogram", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("MFCC", color="white")
    ax.tick_params(colors="white")
    plt.tight_layout()
    return fig


def plot_waveform(y, sr):
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")
    ax.plot(times, y, color="#00BCD4", linewidth=0.8, alpha=0.9)
    ax.fill_between(times, y, alpha=0.2, color="#00BCD4")
    ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.set_title("Waveform", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    plt.tight_layout()
    return fig


def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")

    sorted_idx = np.argsort(probs)
    sorted_names = [f"{EMOTION_EMOJIS[i]} {EMOTION_NAMES[i]}" for i in sorted_idx]
    sorted_probs = probs[sorted_idx]
    sorted_colors = [EMOTION_COLORS[i] for i in sorted_idx]

    bars = ax.barh(sorted_names, sorted_probs * 100, color=sorted_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

    for bar, prob in zip(bars, sorted_probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", ha="left", color="white", fontsize=9)

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="white", fontsize=10)
    ax.set_title("Emotion Probabilities", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", axis="both")
    ax.grid(axis="x", alpha=0.2, color="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    bars[-1].set_edgecolor("white")
    bars[-1].set_linewidth(2)
    plt.tight_layout()
    return fig


def predict_emotion(features: np.ndarray):
    model = load_model()
    X = features[np.newaxis, np.newaxis, ...].astype(np.float32)
    X = _normalize(X)
    X_tensor = torch.from_numpy(X)

    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    pred_class = int(np.argmax(probs))
    return pred_class, probs


def _set_batch_norm_eval(module: nn.Module) -> None:
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.eval()


def fine_tune_from_admin_correction(
    audio_path: str,
    corrected_class_id: int,
    epochs: int = 80,
    learning_rate: float = 1e-3,
):
    if corrected_class_id < 0 or corrected_class_id >= len(EMOTION_NAMES):
        raise ValueError(f"Invalid corrected class id: {corrected_class_id}")

    model = load_model()
    features, _, _, _ = extract_features(audio_path)
    X = features[np.newaxis, np.newaxis, ...].astype(np.float32)
    X = _normalize(X)
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.tensor([corrected_class_id], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        outputs_before = model(X_tensor)
        loss_before = float(criterion(outputs_before, y_tensor).item())
        probs_before = torch.softmax(outputs_before, dim=1)[0]
        pred_before = int(torch.argmax(probs_before).item())
        confidence_before = float(probs_before[pred_before].item())

    for param in model.parameters():
        param.requires_grad = False
    trainable_params = []
    for module in model.classifier:
        if isinstance(module, nn.BatchNorm1d):
            continue
        for param in module.parameters():
            param.requires_grad = True
            trainable_params.append(param)

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)
    safe_epochs = max(1, min(int(epochs), 200))
    epochs_ran = 0
    achieved_target = False

    model.train()
    model.apply(_set_batch_norm_eval)
    for epoch in range(safe_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        epochs_ran = epoch + 1

        with torch.no_grad():
            check_outputs = model(X_tensor)
            check_probs = torch.softmax(check_outputs, dim=1)[0]
            check_pred = int(torch.argmax(check_probs).item())
            check_confidence = float(check_probs[corrected_class_id].item())
            if check_pred == corrected_class_id and check_confidence >= 0.60:
                achieved_target = True
                break

    model.eval()
    with torch.no_grad():
        outputs_after = model(X_tensor)
        loss_after = float(criterion(outputs_after, y_tensor).item())
        probs_after = torch.softmax(outputs_after, dim=1)[0]
        pred_after = int(torch.argmax(probs_after).item())
        confidence_after = float(probs_after[pred_after].item())
        achieved_target = achieved_target or pred_after == corrected_class_id

    output_path = settings.resolve_path(settings.feedback_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(output_path))

    global _model
    _model = model

    for param in model.parameters():
        param.requires_grad = False

    return {
        "loss_before": loss_before,
        "loss_after": loss_after,
        "model_path": str(output_path),
        "epochs": epochs_ran,
        "achieved_target": achieved_target,
        "prediction_before": EMOTION_NAMES[pred_before],
        "confidence_before": confidence_before,
        "prediction_after": EMOTION_NAMES[pred_after],
        "confidence_after": confidence_after,
    }
