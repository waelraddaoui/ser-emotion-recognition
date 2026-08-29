"""
=============================================================================
app.py — Interface de Démonstration Streamlit (RAVDESS SER)
=============================================================================
Titre du PFE : "Classification des Émotions Vocales à partir de MFCCs
               à l'aide de Réseaux Neuronaux Convolutifs :
               Une Étude sur le Corpus RAVDESS"

Interface web interactive pour :
  - Uploader un fichier WAV ou utiliser un exemple du corpus
  - Visualiser le spectrogramme MFCC du fichier
  - Prédire l'émotion avec le modèle CNN entraîné
  - Afficher les probabilités par émotion
=============================================================================
"""

import os
import io
import pickle
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
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

SAMPLE_RATE = 22050
DURATION    = 2.5
OFFSET      = 0.5
N_MFCC      = 40
MAX_FRAMES  = 216

# Labels d'émotions (Ordre numérique RAVDESS 1-8 -> 0-7)
EMOTION_NAMES  = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
EMOTION_EMOJIS = ["😐", "😌", "😊", "😢", "😠", "😨", "🤢", "😲"]
EMOTION_COLORS = ["#607D8B", "#9C27B0", "#FFC107", "#2196F3", "#F44336", "#673AB7", "#4CAF50", "#FF6F00"]

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

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DU MODÈLE (Doit correspondre à train_model_gpu.py)
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
    def forward(self, x): return self.block(x)

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

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE (mis en cache)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, 'best_model_gpu.pth')
    if not os.path.exists(model_path):
        return None
    model = EmotionCNN(n_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model


@st.cache_resource
def load_mlp_model():
    # Ancienne méthode MLP conservée au cas où, mais par défaut on chargera CNN PyTorch
    mlp_path = os.path.join(MODELS_DIR, 'mlp_baseline.pkl')
    if not os.path.exists(mlp_path):
        return None, None
    with open(mlp_path, 'rb') as f:
        model = pickle.load(f)
    return model, None


@st.cache_resource
def load_norm_stats():
    mean_path = os.path.join(MODELS_DIR, 'norm_mean.npy')
    std_path  = os.path.join(MODELS_DIR, 'norm_std.npy')
    if os.path.exists(mean_path) and os.path.exists(std_path):
        m = np.load(mean_path)
        s = np.load(std_path)
        if np.size(m) == 1 and np.size(s) == 1:
            return float(m), float(s)
        return float(np.mean(m)), float(np.mean(s))
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(audio_path):
    """Extrait MFCC 2D + Delta + Delta² avec un chargeur robuste triple cascade."""
    y = None
    sr = SAMPLE_RATE
    
    try:
        if not audio_path or not os.path.exists(audio_path):
            st.error(f"Fichier audio introuvable : {audio_path}")
            return None, None, None, SAMPLE_RATE

        # 1. Tentative avec Soundfile (Le plus précis pour WAV)
        try:
            data, raw_sr = sf.read(audio_path)
            y = data.astype(np.float32)
            if len(y.shape) > 1: y = np.mean(y, axis=1) # Mono
            sr = raw_sr
        except Exception:
            # 2. Tentative avec SciPy (Moteur binairerobuste)
            try:
                raw_sr, data = wavfile.read(audio_path)
                if data.dtype == np.int16: y = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32: y = data.astype(np.float32) / 2147483648.0
                else: y = data.astype(np.float32)
                if len(y.shape) > 1: y = np.mean(y, axis=1)
                sr = raw_sr
            except Exception:
                # 3. Tentative avec Librosa (Dernier recours)
                try:
                    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
                except Exception as e3:
                    st.error(f"Impossible de lire le fichier audio avec les 3 moteurs. Erreur : {e3}")
                    return None, None, None, SAMPLE_RATE

        # Resampling si nécessaire (Uniformisation à 22050 Hz)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # ─── Prétraitement commun ───
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
        st.error(f"Erreur d'extraction finale : {e}")
        return None, None, None, SAMPLE_RATE


def predict_emotion(features, model, norm_mean=None, norm_std=None, is_mlp=False, norm_scaler=None):
    """Applique la normalisation et prédit l'émotion."""
    if features is None:
        return None, None

    if is_mlp:
        # MLP attend un vecteur plat scalé
        X = features.flatten().reshape(1, -1)
        probs = model.predict_proba(X)[0]
        pred_class = np.argmax(probs)
        return pred_class, probs
    else:
        # PyTorch CNN attend (1, 1, 120, 216)
        X = features[np.newaxis, np.newaxis, ...].astype(np.float32)
        if norm_mean is not None and norm_std is not None:
            X = (X - norm_mean) / (norm_std + 1e-6)
        
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()[0]
        
        pred_class = np.argmax(probs)
        return pred_class, probs


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_mfcc_spectrogram(mfcc, sr):
    """Génère un spectrogramme MFCC matplotlib."""
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
    """Génère la forme d'onde du signal audio."""
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig, ax = plt.subplots(figsize=(10, 2))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')
    ax.plot(times, y, color='#00BCD4', linewidth=0.8, alpha=0.9)
    ax.fill_between(times, y, alpha=0.2, color='#00BCD4')
    ax.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Temps (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.set_title('Forme d\'Onde Audio', color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_facecolor('#0f0f23')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    plt.tight_layout()
    return fig


def plot_probabilities(probs):
    """Bar chart horizontal des probabilités par émotion."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#0f0f23')

    sorted_idx = np.argsort(probs)
    sorted_names  = [f"{EMOTION_EMOJIS[i]} {EMOTION_NAMES[i]}" for i in sorted_idx]
    sorted_probs  = probs[sorted_idx]
    sorted_colors = [EMOTION_COLORS[i] for i in sorted_idx]

    bars = ax.barh(sorted_names, sorted_probs * 100, color=sorted_colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

    # Valeur sur chaque barre
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

    # Dernière barre (prédiction) plus marquée
    bars[-1].set_edgecolor('white')
    bars[-1].set_linewidth(2)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TROUVER DES EXEMPLES RAVDESS
# ─────────────────────────────────────────────────────────────────────────────

def find_examples():
    """Retourne un fichier exemple pour chaque émotion."""
    examples = {}
    data_path = os.path.join(BASE_PATH, "data")
    if not os.path.exists(data_path):
        return examples

    for actor_dir in sorted(os.listdir(data_path)):
        actor_path = os.path.join(data_path, actor_dir)
        if not os.path.isdir(actor_path) or not actor_dir.startswith("Actor_"):
            continue
        for fname in sorted(os.listdir(actor_path)):
            if fname.endswith('.wav'):
                parts = fname.split('-')
                if len(parts) >= 7:
                    code = int(parts[2])
                    if code in RAVDESS_MAPPING:
                        idx = RAVDESS_MAPPING[code]
                        if idx not in examples:
                            examples[idx] = os.path.join(actor_path, fname)
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Config page ───────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="SER — Reconnaissance d'Émotions Vocales",
        page_icon="🎙️",
        layout="wide"
    )

    # ── CSS personnalisé (thème sombre premium) ───────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
        color: white;
    }
    .hero-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00BCD4, #9C27B0, #FF5722);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #90CAF9;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .emotion-emoji {
        font-size: 4rem;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .emotion-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .confidence-badge {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 1rem;
        display: inline-block;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .info-box {
        background: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196F3;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        color: #90CAF9;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 24px;
        width: 100%;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── En-tête ───────────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">🎙️ Reconnaissance d\'Émotions Vocales</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Classification des Émotions Vocales à partir de MFCCs '
        'à l\'aide de Réseaux Neuronaux Convolutifs · Corpus RAVDESS · Master Data Science</div>',
        unsafe_allow_html=True
    )

    # ── Chargement des modèles ───────────────────────────────────────────────
    cnn_model = load_model()
    mlp_model, mlp_scaler = load_mlp_model()
    norm_mean, norm_std = load_norm_stats()

    if cnn_model is None and mlp_model is None:
        st.error(
            "⚠️ Aucun modèle trouvé (CNN ou MLP). Veuillez d'abord lancer l'entraînement."
        )
        return

    # ── Sidebar : Paramètres du modèle ───────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤖 Choix du Modèle")
        model_type = st.radio(
            "Sélectionnez le moteur d'analyse :",
            ["CNN (PyTorch 97.12% - 8 Émotions)", "MLP Baseline"],
            index=0,
            help="Le nouveau modèle CNN-5 est désormais le plus performant."
        )
        is_mlp = "MLP" in model_type
        current_model = mlp_model if is_mlp else cnn_model
        
        status_color = "#4CAF50" if current_model else "#F44336"
        st.markdown(f"""
        <div style="padding:10px; border-radius:5px; background: {status_color}20; border: 1px solid {status_color}; text-align:center;">
            Mode : <strong>{'MLP' if is_mlp else 'CNN'}</strong> actif
        </div>
        """, unsafe_allow_html=True)

    # ── Mise en page principale ───────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown("### 📂 Source Audio")

        tab_upload, tab_example = st.tabs(["📁 Upload WAV", "🎭 Exemple RAVDESS"])

        audio_path = None

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Déposez un fichier audio WAV ici",
                type=['wav', 'mp3', 'ogg'],
                help="Format recommandé : WAV, 16-bit, 22050 Hz. Durée idéale : 3-5 secondes."
            )
            if uploaded_file:
                # On ne recrée le fichier que s'il est différent ou absent
                if st.session_state.get('last_uploaded_name') != uploaded_file.name:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        audio_path = tmp.name
                        st.session_state['audio_path'] = audio_path
                        st.session_state['last_uploaded_name'] = uploaded_file.name
                else:
                    audio_path = st.session_state.get('audio_path')
                
                # Petit bloc d'info sur le fichier
                st.audio(uploaded_file)
                st.markdown(f"""
                <div class="info-box" style="background: rgba(0, 188, 212, 0.1); border-color: #00BCD4;">
                    🎯 <strong>Fichier chargé avec succès :</strong> {uploaded_file.name}<br>
                    📏 <strong>Taille :</strong> {uploaded_file.size / 1024:.1f} KB<br>
                    💡 <em>Le signal sera traité via soundfile pour maximiser la compatibilité.</em>
                </div>
                """, unsafe_allow_html=True)

        with tab_example:
            examples = find_examples()
            if examples:
                selected_emotion = st.selectbox(
                    "Choisir une émotion",
                    options=list(range(len(EMOTION_NAMES))),
                    format_func=lambda i: f"{EMOTION_EMOJIS[i]} {EMOTION_NAMES[i]}"
                )
                if st.button("🎵 Charger cet exemple", key="load_example"):
                    if selected_emotion in examples:
                        audio_path = examples[selected_emotion]
                        st.audio(audio_path)
                        st.session_state['audio_path'] = audio_path
            else:
                st.warning("Dossier `data/` introuvable.")

        if 'audio_path' in st.session_state and audio_path is None:
            audio_path = st.session_state.get('audio_path')

        # ── Bouton Analyser ───────────────────────────────────────────────────
        st.markdown("---")
        analyze = st.button("🔍 Analyser l'Émotion", key="analyze_btn")

    # ── Résultats ─────────────────────────────────────────────────────────────
    with col_right:
        if analyze and audio_path:
            with st.spinner("Extraction des features MFCC..."):
                try:
                    features, mfcc, y_audio, sr = extract_features(audio_path)
                    
                    if features is None or y_audio is None:
                        st.error("❌ Impossible d'extraire les caractéristiques. Le format du fichier audio est peut-être corrompu ou non supporté.")
                        return
                    
                    # Vérification de la "santé" du signal (après le check None)
                    rms = np.sqrt(np.mean(y_audio**2))
                    if rms < 0.01:
                        st.warning("⚠️ **Signal très faible détecté.** Votre voix est peut-être trop loin du micro ou masquée par le silence.")
                    
                    pred_class, probs = predict_emotion(features, current_model, norm_mean, norm_std, is_mlp=is_mlp, norm_scaler=mlp_scaler)
                    
                    if pred_class is None:
                        st.error("❌ Erreur lors de la prédiction. Réessayez avec un autre fichier.")
                        return

                    confidence = probs[pred_class]
                    emotion    = EMOTION_NAMES[pred_class]
                    emoji      = EMOTION_EMOJIS[pred_class]
                    color      = EMOTION_COLORS[pred_class]

                    # ── Carte résultat principal ──────────────────────────────
                    st.markdown(f"""
                    <div class="emotion-card" style="border-color:{color}40; background: {color}15;">
                        <div class="emotion-emoji">{emoji}</div>
                        <div class="emotion-label" style="color:{color}">{emotion}</div>
                        <div class="confidence-badge">Confiance : <strong>{confidence*100:.1f}%</strong></div>
                        <p style="color:#aaa; margin-top:12px; font-size:0.9rem;">
                            {EMOTION_DESCRIPTIONS[emotion]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")

                    # ── Probabilités ──────────────────────────────────────────
                    st.markdown("#### 📊 Distribution de Confiance")
                    fig_prob = plot_probabilities(probs)
                    st.pyplot(fig_prob, use_container_width=True)
                    plt.close()

                    # ── Spectrogramme MFCC ────────────────────────────────────
                    st.markdown("#### 🌊 Spectrogramme MFCC")
                    fig_mfcc = plot_mfcc_spectrogram(mfcc, sr)
                    st.pyplot(fig_mfcc, use_container_width=True)
                    plt.close()

                    # ── Waveform ──────────────────────────────────────────────
                    st.markdown("#### 📈 Forme d'Onde")
                    fig_wave = plot_waveform(y_audio, sr)
                    st.pyplot(fig_wave, use_container_width=True)
                    plt.close()

                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse : {e}")
                    st.exception(e)

        elif not analyze:
            # Affichage de bienvenue si pas encore d'analyse
            st.markdown("### 🎯 Comment utiliser cette interface")
            st.markdown("""
            <div class="info-box">
                1. <strong>Chargez un fichier WAV</strong> depuis votre ordinateur, ou<br>
                2. <strong>Sélectionnez un exemple</strong> du corpus RAVDESS<br>
                3. Cliquez sur <strong>Analyser l'Émotion</strong>
            </div>
            """, unsafe_allow_html=True)

            # Grille des émotions disponibles
            st.markdown("### 🎭 Émotions Classifiées")
            cols = st.columns(4)
            for i, (name, emoji, color) in enumerate(zip(EMOTION_NAMES, EMOTION_EMOJIS, EMOTION_COLORS)):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color:{color}40; margin-bottom:10px;">
                        <div style="font-size:2rem">{emoji}</div>
                        <div style="color:{color}; font-weight:600">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Informations du projet (sidebar) ──────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎓 À Propos du Projet")
        st.markdown("""
        **Titre :** Classification des Émotions Vocales à partir de MFCCs à l'aide de Réseaux Neuronaux Convolutifs

        **Corpus :** RAVDESS — 24 acteurs, 8 émotions

        **Features :** MFCC + Δ + ΔΔ → tenseur (120×216×1)

        **Modèle :** CNN 5 blocs profond (LeakyReLU)

        **Framework :** PyTorch (GPU Optimized)
        """)
        st.divider()
        with st.expander("💡 Conseils d'Enregistrement"):
            st.markdown("""
            Pour maximiser la précision avec vos propres fichiers :
            1. **Environnement Calme** : Évitez les bruits de fond (ventilo, rue).
            2. **Micro Proche** : Parlez à ~15cm du microphone.
            3. **Expression Nette** : Exagérez légèrement l'émotion pour un meilleur contraste.
            4. **Durée** : Essayez de parler pendant au moins 2 à 3 secondes.
            """)
        st.divider()
        st.markdown("*Travail de Fin d'Études — DS 2026*")


def _show_project_info():
    """Affiche les infos du projet quand le modèle n'est pas encore entraîné."""
    st.markdown("---")
    st.markdown("## 🏗️ Architecture du Projet")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Pipeline complet :**
        1. 📁 Dataset RAVDESS (WAV)
        2. 🔧 `prepare_data.py` — Extraction MFCC 2D
        3. 🤖 `train_model.py` — CNN
        4. 📊 `evaluate_model.py` — Métriques
        5. 📈 `visualize.py` — Figures
        6. 🌐 `app.py` — Cette interface
        """)
    with col2:
        st.markdown("""
        **Émotions classifiées :**
        """)
        for emoji, name in zip(EMOTION_EMOJIS, EMOTION_NAMES):
            st.markdown(f"{emoji} {name}")


if __name__ == "__main__":
    main()
