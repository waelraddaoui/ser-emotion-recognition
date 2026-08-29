import json

en_file = 'en.json'
fr_file = 'fr.json'

with open(en_file, 'r', encoding='utf-8') as f:
    en_data = json.load(f)

with open(fr_file, 'r', encoding='utf-8') as f:
    fr_data = json.load(f)

en_additions = {
  "STUDIO": {
    "TITLE": "🎙️ SER AI Studio",
    "SUBTITLE": "Deep CNN-5 Speech Emotion Recognition — 8 Emotions (97.12%)",
    "TAB_UPLOAD": "📁 Upload Audio",
    "TAB_RECORD": "🎤 Record",
    "TAB_EXAMPLE": "🎭 RAVDESS Example",
    "RECORD_TITLE": "🎤 Real-time Voice Recording",
    "START_RECORD": "🎙️ Start Recording",
    "PAUSE": "⏸️ Pause",
    "RESUME": "▶️ Resume",
    "STOP": "⏹️ Stop",
    "RECORDED_AUDIO": "🔊 Recorded Audio",
    "RE_RECORD": "✏️ Re-record",
    "ANALYZE": "🔍 Analyze",
    "ANALYZING": "Analyzing...",
    "AUDIO_CHECK": "Audio check...",
    "UPLOAD_TITLE": "📁 Analyze Audio",
    "UPLOAD_DROP": "Drop your audio file",
    "UPLOAD_HINT": "Or click to browse (Max 5s, WAV/MP3/FLAC/M4A/OGG)",
    "EXAMPLE_TITLE": "🎭 RAVDESS Examples",
    "QUALITY_TITLE": "Audio quality gate",
    "QUALITY_PREFLIGHT": "Preflight check before prediction",
    "QUALITY_SCORE": "Score",
    "QUALITY_SILENCE": "Silence",
    "QUALITY_CLIPPING": "Clipping",
    "QUALITY_PEAK": "Peak",
    "WARNINGS": "Warnings",
    "RECOMMENDATIONS": "Recommendations",
    "CLEAN_RECORDING": "Clean recording",
    "NO_MAJOR_ISSUE": "No major issue detected before prediction.",
    "CONFIDENCE_DIST": "📊 Confidence Distribution",
    "MFCC": "🌊 MFCC Spectrogram",
    "WAVEFORM": "📈 Waveform",
    "EXPLAINABILITY_TITLE": "How the model ranked the emotions",
    "DECISION_GAP": "Decision gap",
    "REPORT_ERROR_TITLE": "Report an error",
    "REPORT_ERROR_DESC": "If this prediction is incorrect, report it to the admin for verification.",
    "READY_SEND": "Ready to send",
    "SECURE": "Secure",
    "REASON": "Reason",
    "COMMENT": "Comment",
    "SEND_ADMIN": "Send to admin",
    "REASONS": {
      "WRONG_EMOTION": "Wrong emotion detected",
      "LOW_QUALITY": "Poor audio quality",
      "NOISE": "Too much noise",
      "OTHER": "Other"
    }
  },
  "ADMIN": {
    "TITLE": "📊 Admin Data Science Dashboard",
    "SUBTITLE": "Real-time analytics, model performance, and user management",
    "TAB_DASHBOARD": "📈 Dashboard",
    "TAB_EXPLORER": "🔍 Explorer",
    "TAB_PERFORMANCE": "📊 Performance",
    "TAB_REPORTS": "🚩 Reports",
    "TAB_USERS": "👥 Users",
    "TAB_TRAINING": "Training",
    "TOTAL_PREDICTIONS": "Total Predictions",
    "ACTIVE_USERS": "Active Users",
    "AVG_CONFIDENCE": "Avg Confidence",
    "EMOTION_DIST": "🎭 Emotion Distribution",
    "CONFIDENCE_METRICS": "📊 Confidence Metrics",
    "EXPLORER_TITLE": "🔍 Prediction Explorer",
    "FILTER_EMOTION": "Filter by Emotion",
    "ALL_EMOTIONS": "All Emotions",
    "MIN_CONFIDENCE": "Min Confidence",
    "EXPORT_CSV": "📥 Export CSV",
    "NO_PREDICTIONS": "No predictions found",
    "SUPERVISED_CURATION": "Supervised dataset curation",
    "TRUE_EMOTION": "True emotion",
    "SAVE_TRAINING": "Save to training folder",
    "LOW_CONFIDENCE": "⚠️ Low Confidence Predictions (< 60%)",
    "RECOMMENDATIONS": "💡 Recommendations",
    "USER_REPORTS": "🚩 User Reports",
    "FILTER_STATUS": "Filter by Status",
    "REPORT_AUDIO": "Reported vocal",
    "ADMIN_NOTE": "Admin review note",
    "SUPERVISED_CORRECTION": "Supervised correction",
    "MARK_FAULT": "Mark fault",
    "REJECT": "Reject",
    "RETEST": "Re-test model",
    "TEACH_MODEL": "Teach model",
    "USER_MANAGEMENT": "👥 User Management",
    "ADD_USER": "+ Add User",
    "MODEL_TRAINING": "Model Training",
    "START_TRAINING": "Start training"
  }
}

fr_additions = {
  "STUDIO": {
    "TITLE": "🎙️ SER AI Studio",
    "SUBTITLE": "Reconnaissance d'Emotions Vocales par CNN-5 Profond — 8 Émotions (97.12%)",
    "TAB_UPLOAD": "📁 Analyser Audio",
    "TAB_RECORD": "🎤 Enregistrer",
    "TAB_EXAMPLE": "🎭 Exemple RAVDESS",
    "RECORD_TITLE": "🎤 Enregistrement Vocal en Temps Réel",
    "START_RECORD": "🎙️ Démarrer l'enregistrement",
    "PAUSE": "⏸️ Pause",
    "RESUME": "▶️ Reprendre",
    "STOP": "⏹️ Arrêter",
    "RECORDED_AUDIO": "🔊 Audio enregistré",
    "RE_RECORD": "✏️ Ré-enregistrer",
    "ANALYZE": "🔍 Analyser",
    "ANALYZING": "Analyse en cours...",
    "AUDIO_CHECK": "Contrôle audio...",
    "UPLOAD_TITLE": "📁 Analyser un Audio",
    "UPLOAD_DROP": "Déposez votre fichier audio",
    "UPLOAD_HINT": "Ou cliquez pour parcourir (Max 5s, WAV/MP3/FLAC/M4A/OGG)",
    "EXAMPLE_TITLE": "🎭 Exemples RAVDESS",
    "QUALITY_TITLE": "Contrôle audio",
    "QUALITY_PREFLIGHT": "Vérification avant prédiction",
    "QUALITY_SCORE": "Score",
    "QUALITY_SILENCE": "Silence",
    "QUALITY_CLIPPING": "Sursaturation",
    "QUALITY_PEAK": "Crête",
    "WARNINGS": "Avertissements",
    "RECOMMENDATIONS": "Recommandations",
    "CLEAN_RECORDING": "Enregistrement propre",
    "NO_MAJOR_ISSUE": "Aucun problème majeur détecté avant prédiction.",
    "CONFIDENCE_DIST": "📊 Distribution de Confiance",
    "MFCC": "🌊 Spectrogramme MFCC",
    "WAVEFORM": "📈 Forme d'Onde",
    "EXPLAINABILITY_TITLE": "Comment le modèle a classé les émotions",
    "DECISION_GAP": "Écart de décision",
    "REPORT_ERROR_TITLE": "Signaler une erreur",
    "REPORT_ERROR_DESC": "Si cette prédiction est incorrecte, transmettez-la à l'admin pour vérification.",
    "READY_SEND": "Prêt à envoyer",
    "SECURE": "Sécurisé",
    "REASON": "Motif",
    "COMMENT": "Commentaire",
    "SEND_ADMIN": "Envoyer à l'admin",
    "REASONS": {
      "WRONG_EMOTION": "Mauvaise émotion détectée",
      "LOW_QUALITY": "Audio de mauvaise qualité",
      "NOISE": "Trop de bruit / micro",
      "OTHER": "Autre"
    }
  },
  "ADMIN": {
    "TITLE": "📊 Dashboard Admin Data Science",
    "SUBTITLE": "Analytique en temps réel, performance et utilisateurs",
    "TAB_DASHBOARD": "📈 Dashboard",
    "TAB_EXPLORER": "🔍 Explorateur",
    "TAB_PERFORMANCE": "📊 Performance",
    "TAB_REPORTS": "🚩 Signalements",
    "TAB_USERS": "👥 Utilisateurs",
    "TAB_TRAINING": "Entraînement",
    "TOTAL_PREDICTIONS": "Prédictions totales",
    "ACTIVE_USERS": "Utilisateurs actifs",
    "AVG_CONFIDENCE": "Confiance moy.",
    "EMOTION_DIST": "🎭 Distribution d'Émotions",
    "CONFIDENCE_METRICS": "📊 Métriques de Confiance",
    "EXPLORER_TITLE": "🔍 Explorateur de Prédictions",
    "FILTER_EMOTION": "Filtrer par émotion",
    "ALL_EMOTIONS": "Toutes les émotions",
    "MIN_CONFIDENCE": "Confiance min",
    "EXPORT_CSV": "📥 Exporter CSV",
    "NO_PREDICTIONS": "Aucune prédiction trouvée",
    "SUPERVISED_CURATION": "Curation supervisée",
    "TRUE_EMOTION": "Vraie émotion",
    "SAVE_TRAINING": "Sauvegarder pour l'entraînement",
    "LOW_CONFIDENCE": "⚠️ Prédictions basse confiance (< 60%)",
    "RECOMMENDATIONS": "💡 Recommandations",
    "USER_REPORTS": "🚩 Signalements utilisateurs",
    "FILTER_STATUS": "Filtrer par statut",
    "REPORT_AUDIO": "Audio signalé",
    "ADMIN_NOTE": "Note de l'admin",
    "SUPERVISED_CORRECTION": "Correction supervisée",
    "MARK_FAULT": "Marquer comme faux",
    "REJECT": "Rejeter",
    "RETEST": "Re-tester le modèle",
    "TEACH_MODEL": "Entraîner le modèle",
    "USER_MANAGEMENT": "👥 Gestion des Utilisateurs",
    "ADD_USER": "+ Ajouter Utilisateur",
    "MODEL_TRAINING": "Entraînement du modèle",
    "START_TRAINING": "Démarrer l'entraînement"
  }
}

en_data.update(en_additions)
fr_data.update(fr_additions)

with open(en_file, 'w', encoding='utf-8') as f:
    json.dump(en_data, f, indent=2, ensure_ascii=False)

with open(fr_file, 'w', encoding='utf-8') as f:
    json.dump(fr_data, f, indent=2, ensure_ascii=False)

print("Translations updated successfully.")
