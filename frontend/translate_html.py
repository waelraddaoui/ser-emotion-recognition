import os

def apply_translations():
    # Fix ser.component.html
    ser_path = r'src\app\ser\ser.component.html'
    with open(ser_path, 'r', encoding='utf-8') as f:
        ser = f.read()
    
    # Replace hardcoded text with translation keys
    replacements_ser = {
        '🎙️ SER AI Studio': "{{ 'STUDIO.TITLE' | translate }}",
        "Reconnaissance d'Emotions Vocales par CNN-5 Profond — 8 Émotions (97.12%)": "{{ 'STUDIO.SUBTITLE' | translate }}",
        "📁 Analyser Audio": "{{ 'STUDIO.TAB_UPLOAD' | translate }}",
        "🎤 Enregistrer": "{{ 'STUDIO.TAB_RECORD' | translate }}",
        "🎭 Exemple RAVDESS": "{{ 'STUDIO.TAB_EXAMPLE' | translate }}",
        "🎤 Enregistrement Vocal en Temps Réel": "{{ 'STUDIO.RECORD_TITLE' | translate }}",
        "🎙️ Démarrer l'enregistrement": "{{ 'STUDIO.START_RECORD' | translate }}",
        "⏸️ Pause": "{{ 'STUDIO.PAUSE' | translate }}",
        "▶️ Reprendre": "{{ 'STUDIO.RESUME' | translate }}",
        "⏹️ Arrêter": "{{ 'STUDIO.STOP' | translate }}",
        "🔊 Audio enregistré :": "{{ 'STUDIO.RECORDED_AUDIO' | translate }}",
        "✏️ Ré-enregistrer": "{{ 'STUDIO.RE_RECORD' | translate }}",
        "🔍 Analyser": "{{ 'STUDIO.ANALYZE' | translate }}",
        "Analyse en cours...": "{{ 'STUDIO.ANALYZING' | translate }}",
        "📁 Analyser un Audio": "{{ 'STUDIO.UPLOAD_TITLE' | translate }}",
        "Déposez votre fichier audio": "{{ 'STUDIO.UPLOAD_DROP' | translate }}",
        "🎭 Exemples RAVDESS": "{{ 'STUDIO.EXAMPLE_TITLE' | translate }}",
        "Contrôle audio": "{{ 'STUDIO.QUALITY_TITLE' | translate }}",
        "Vérification avant prédiction": "{{ 'STUDIO.QUALITY_PREFLIGHT' | translate }}",
        "Avertissements": "{{ 'STUDIO.WARNINGS' | translate }}",
        "Recommandations": "{{ 'STUDIO.RECOMMENDATIONS' | translate }}",
        "Enregistrement propre": "{{ 'STUDIO.CLEAN_RECORDING' | translate }}",
        "Aucun problème majeur détecté avant prédiction.": "{{ 'STUDIO.NO_MAJOR_ISSUE' | translate }}",
        "📊 Distribution de Confiance": "{{ 'STUDIO.CONFIDENCE_DIST' | translate }}",
        "🌊 Spectrogramme MFCC": "{{ 'STUDIO.MFCC' | translate }}",
        "📈 Forme d'Onde": "{{ 'STUDIO.WAVEFORM' | translate }}",
        "Comment le modèle a classé les émotions": "{{ 'STUDIO.EXPLAINABILITY_TITLE' | translate }}",
        "Écart de décision :": "{{ 'STUDIO.DECISION_GAP' | translate }}",
        "Signaler une erreur": "{{ 'STUDIO.REPORT_ERROR_TITLE' | translate }}",
        "Si cette prédiction est incorrecte, transmettez-la à l'admin pour vérification.": "{{ 'STUDIO.REPORT_ERROR_DESC' | translate }}",
        "Motif :": "{{ 'STUDIO.REASON' | translate }}",
        "Commentaire :": "{{ 'STUDIO.COMMENT' | translate }}",
        "Envoyer à l'admin": "{{ 'STUDIO.SEND_ADMIN' | translate }}"
    }
    
    for old, new in replacements_ser.items():
        ser = ser.replace(old, new)
        
    with open(ser_path, 'w', encoding='utf-8') as f:
        f.write(ser)

    # Fix admin-dashboard.component.html
    admin_path = r'src\app\admin\admin-dashboard.component.html'
    with open(admin_path, 'r', encoding='utf-8') as f:
        admin = f.read()
        
    replacements_admin = {
        "🛡️ Admin Dashboard": "{{ 'ADMIN.TITLE' | translate }}",
        "Real-time analytics, model performance, and user management": "{{ 'ADMIN.SUBTITLE' | translate }}",
        "📈 Dashboard": "{{ 'ADMIN.TAB_DASHBOARD' | translate }}",
        "🔍 Explorateur": "{{ 'ADMIN.TAB_EXPLORER' | translate }}",
        "📊 Performance": "{{ 'ADMIN.TAB_PERFORMANCE' | translate }}",
        "🚩 Signalements": "{{ 'ADMIN.TAB_REPORTS' | translate }}",
        "👥 Utilisateurs": "{{ 'ADMIN.TAB_USERS' | translate }}",
        "Entraînement": "{{ 'ADMIN.TAB_TRAINING' | translate }}",
        "Prédictions totales": "{{ 'ADMIN.TOTAL_PREDICTIONS' | translate }}",
        "Utilisateurs actifs": "{{ 'ADMIN.ACTIVE_USERS' | translate }}",
        "Confiance moy.": "{{ 'ADMIN.AVG_CONFIDENCE' | translate }}",
        "🎭 Distribution d'Émotions": "{{ 'ADMIN.EMOTION_DIST' | translate }}",
        "📊 Métriques de Confiance": "{{ 'ADMIN.CONFIDENCE_METRICS' | translate }}",
        "🔍 Explorateur de Prédictions": "{{ 'ADMIN.EXPLORER_TITLE' | translate }}",
        "Filtrer par émotion...": "{{ 'ADMIN.FILTER_EMOTION' | translate }}",
        "Toutes les émotions": "{{ 'ADMIN.ALL_EMOTIONS' | translate }}",
        "Confiance min:": "{{ 'ADMIN.MIN_CONFIDENCE' | translate }}",
        "📥 Exporter CSV": "{{ 'ADMIN.EXPORT_CSV' | translate }}",
        "Aucune prédiction trouvée": "{{ 'ADMIN.NO_PREDICTIONS' | translate }}",
        "Curation supervisée": "{{ 'ADMIN.SUPERVISED_CURATION' | translate }}",
        "Vraie émotion:": "{{ 'ADMIN.TRUE_EMOTION' | translate }}",
        "Sauvegarder pour l'entraînement": "{{ 'ADMIN.SAVE_TRAINING' | translate }}",
        "⚠️ Prédictions basse confiance (< 60%)": "{{ 'ADMIN.LOW_CONFIDENCE' | translate }}",
        "💡 Recommandations": "{{ 'ADMIN.RECOMMENDATIONS' | translate }}",
        "🚩 Signalements utilisateurs": "{{ 'ADMIN.USER_REPORTS' | translate }}",
        "Filtrer par statut...": "{{ 'ADMIN.FILTER_STATUS' | translate }}",
        "Audio signalé": "{{ 'ADMIN.REPORT_AUDIO' | translate }}",
        "Note de l'admin:": "{{ 'ADMIN.ADMIN_NOTE' | translate }}",
        "Correction supervisée": "{{ 'ADMIN.SUPERVISED_CORRECTION' | translate }}",
        "Marquer comme faux": "{{ 'ADMIN.MARK_FAULT' | translate }}",
        "Rejeter": "{{ 'ADMIN.REJECT' | translate }}",
        "Re-tester le modèle": "{{ 'ADMIN.RETEST' | translate }}",
        "Entraîner le modèle": "{{ 'ADMIN.TEACH_MODEL' | translate }}",
        "👥 Gestion des Utilisateurs": "{{ 'ADMIN.USER_MANAGEMENT' | translate }}",
        "+ Ajouter Utilisateur": "{{ 'ADMIN.ADD_USER' | translate }}",
        "Entraînement du modèle": "{{ 'ADMIN.MODEL_TRAINING' | translate }}",
        "Démarrer l'entraînement": "{{ 'ADMIN.START_TRAINING' | translate }}"
    }
    
    for old, new in replacements_admin.items():
        admin = admin.replace(old, new)
        
    with open(admin_path, 'w', encoding='utf-8') as f:
        f.write(admin)
        
    print("Translations applied successfully.")

apply_translations()
