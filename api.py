import os
import io
import pickle
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
from scipy.io import wavfile
import soundfile as sf
import tempfile
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_PATH, "models")
DATA_PATH  = os.path.join(BASE_PATH, "data")

SAMPLE_RATE = 22050
DURATION    = 2.5
OFFSET      = 0.5
N_MFCC      = 40
MAX_FRAMES  = 216

# Labels d'émotions (Ordre numérique RAVDESS 1-8 -> 0-7)
EMOTION_NAMES  = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
EMOTION_EMOJIS = ["😐", "😌", "😊", "😢", "😠", "😨", "🤢", "😲"]
EMOTION_COLORS = ["#607D8B", "#9C27B0", "#FFC107", "#2196F3", "#F44336", "#673AB7", "#4CAF50", "#FF6F00"]
EMOTION_DESCRIPTIONS = {
    "Neutral":  "Ton de voix neutre, sans expression émotionnelle marquée.",
    "Calm":     "Expression de sérénité et de tranquillité. Voix posée et équilibrée.",
    "Happy":    "Ton joyeux, enthousiaste et dynamique, exprimant la joie.",
    "Sad":      "Expression de tristesse, voix souvent plus lente et mélancolique.",
    "Angry":    "Ton agressif, fort et tendu, exprimant la colère.",
    "Fearful":  "Expression d'anxiété, voix tremblante trahissant la peur.",
    "Disgust":  "Ton de dégoût ou de répulsion face à quelque chose de désagréable.",
    "Surprised": "Ton surpris ou étonné, marqué par une intonation soudaine et élevée."
}

# Mapping RAVDESS
RAVDESS_MAPPING = {
    1: 0, # Neutral
    2: 1, # Calm
    3: 2, # Happy
    4: 3, # Sad
    5: 4, # Angry
    6: 5, # Fearful
    7: 6, # Disgust
    8: 7  # Surprised
}

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DU MODÈLE PyTorch
# ─────────────────────────────────────────────────────────────────────────────

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
            ConvBlock(1, 32,   dropout=0.15),
            ConvBlock(32, 64,  dropout=0.20),
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
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ── Chargement au démarrage de l'API
model_path = os.path.join(MODELS_DIR, 'best_model_gpu.pth')
model = EmotionCNN(n_classes=8)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

mean_path = os.path.join(MODELS_DIR, 'norm_mean.npy')
std_path  = os.path.join(MODELS_DIR, 'norm_std.npy')
norm_mean, norm_std = None, None
if os.path.exists(mean_path) and os.path.exists(std_path):
    norm_mean = float(np.load(mean_path))
    norm_std  = float(np.load(std_path))

# ─────────────────────────────────────────────────────────────────────────────
# LOGIQUE D'EXTRACTION (Triple Cascade)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(audio_path):
    y = None
    sr = SAMPLE_RATE
    try:
        if not audio_path or not os.path.exists(audio_path):
            raise Exception(f"Fichier audio introuvable : {audio_path}")
        
        try:
            data, raw_sr = sf.read(audio_path)
            y = data.astype(np.float32)
            if len(y.shape) > 1: y = np.mean(y, axis=1) # Mono
            sr = raw_sr
        except Exception:
            try:
                raw_sr, data = wavfile.read(audio_path)
                if data.dtype == np.int16: y = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32: y = data.astype(np.float32) / 2147483648.0
                else: y = data.astype(np.float32)
                if len(y.shape) > 1: y = np.mean(y, axis=1)
                sr = raw_sr
            except Exception:
                try:
                    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                except Exception as e3:
                    raise Exception(f"Impossible de lire le fichier audio avec les 3 moteurs. Erreur : {e3}")

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
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        
        if combined.shape[1] < MAX_FRAMES:
            pad_w = MAX_FRAMES - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_w)), mode='constant')
        else:
            combined = combined[:, :MAX_FRAMES]

        return combined, mfcc, y_final, sr
    except Exception as e:
        print(f"Erreur extraction: {e}")
        return None, None, None, None

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def plot_mfcc_spectrogram(mfcc, sr):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax, cmap='magma')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogramme MFCC (40 coefficients)', color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Temps (s)', color='white')
    ax.set_ylabel('Coefficients MFCC', color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    return fig

def plot_waveform(y, sr):
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    ax.plot(times, y, color='#00BCD4', linewidth=0.8, alpha=0.9)
    ax.fill_between(times, y, alpha=0.2, color='#00BCD4')
    ax.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Temps (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.set_title("Forme d'Onde Audio", color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    plt.tight_layout()
    return fig

def plot_probabilities(probs):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')

    sorted_idx = np.argsort(probs)
    sorted_names  = [f"{EMOTION_EMOJIS[i]} {EMOTION_NAMES[i]}" for i in sorted_idx]
    sorted_probs  = probs[sorted_idx]
    sorted_colors = [EMOTION_COLORS[i] for i in sorted_idx]

    bars = ax.barh(sorted_names, sorted_probs * 100, color=sorted_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

    for bar, prob in zip(bars, sorted_probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', ha='left', color='white', fontsize=9)

    ax.set_xlim(0, 115)
    ax.set_xlabel('Confiance (%)', color='white', fontsize=10)
    ax.set_title('Probabilités par Émotion', color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white', axis='both')
    ax.grid(axis='x', alpha=0.2, color='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    bars[-1].set_edgecolor('white')
    bars[-1].set_linewidth(2)
    plt.tight_layout()
    return fig

def predict_emotion(features, mfcc, y_audio, sr):
    if features is None:
        return None
    X = features[np.newaxis, np.newaxis, ...].astype(np.float32)
    if norm_mean is not None and norm_std is not None:
        X = (X - norm_mean) / (norm_std + 1e-6)
    
    X_tensor = torch.from_numpy(X)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
    
    pred_class = int(np.argmax(probs))
    
    return {
        "class_id": pred_class,
        "emotion": EMOTION_NAMES[pred_class],
        "emoji": EMOTION_EMOJIS[pred_class],
        "color": EMOTION_COLORS[pred_class],
        "confidence": float(probs[pred_class]),
        "description": EMOTION_DESCRIPTIONS[EMOTION_NAMES[pred_class]],
        "probabilities": [float(p) for p in probs],
        "img_prob": fig_to_base64(plot_probabilities(probs)),
        "img_mfcc": fig_to_base64(plot_mfcc_spectrogram(mfcc, sr)),
        "img_wave": fig_to_base64(plot_waveform(y_audio, sr))
    }

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="SER PFE API", description="API pour la reconnaissance d'émotions vocales")

import uuid

# Cors pour autoriser Angular à appeler l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "API SER en ligne", "model_loaded": model is not None}

@app.get("/api/audio")
def get_audio(filepath: str):
    if os.path.exists(filepath) and filepath.endswith('.wav'):
        return FileResponse(filepath, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio non trouvé")

@app.get("/api/examples")
def get_examples():
    if not os.path.exists(DATA_PATH):
        return []
    
    examples = []
    # Parcourir Actor_01 pour simplifier, on prendra un exemple par émotion
    actor_path = os.path.join(DATA_PATH, "Actor_01")
    if os.path.exists(actor_path):
        for fname in os.listdir(actor_path):
            if fname.endswith('.wav'):
                parts = fname.split('-')
                if len(parts) >= 7:
                    code = int(parts[2])
                    if code in RAVDESS_MAPPING:
                        idx = RAVDESS_MAPPING[code]
                        if not any(e["id"] == idx for e in examples):
                            # Replace backslashes for URL compatibility
                            f_path = os.path.join(actor_path, fname).replace('\\', '/')
                            examples.append({
                                "id": idx,
                                "filename": fname,
                                "emotion": EMOTION_NAMES[idx],
                                "emoji": EMOTION_EMOJIS[idx],
                                "audio_url": f"/audio?filepath={f_path}"
                            })
    return sorted(examples, key=lambda x: x["id"])


class ExampleRequest(BaseModel):
    example_id: int

@app.post("/api/predict/example")
def predict_from_example(req: ExampleRequest):
    # Retrouver le filepath à partir de l'example_id
    examples = get_examples()
    filepath = None
    for ex in examples:
        if ex["id"] == req.example_id:
            filepath = ex["audio_url"].split("filepath=")[1]
            break
            
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    features, mfcc, y_audio, sr = extract_features(filepath)
    if features is None:
        raise HTTPException(status_code=500, detail="Echec extraction features")
    result = predict_emotion(features, mfcc, y_audio, sr)
    return {"success": True, "result": result}

@app.post("/api/predict/upload")
async def predict_from_upload(file: UploadFile = File(...)):
    temp_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}.wav")
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'écriture: {e}")

    features, mfcc, y_audio, sr = extract_features(temp_path)
    os.remove(temp_path) # Nettoyage

    if features is None:
        raise HTTPException(status_code=500, detail="Echec extraction features")
    result = predict_emotion(features, mfcc, y_audio, sr)
    return {"success": True, "result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
