
"""
Génération du rapport de PFE — Wael Raddaoui
Titre : Classification des Émotions Vocales à partir de MFCCs
        à l'aide de Réseaux Neuronaux Convolutifs : Une Étude sur le Corpus RAVDESS
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_PATH, "figures")
CAPTUREPHONE_DIR = os.path.join(BASE_PATH, "capturesphone")

doc = Document()

# ─── Marges ───────────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Cm(2.5)
section.bottom_margin = Cm(2.5)
section.left_margin   = Cm(3.0)
section.right_margin  = Cm(2.5)

# ─── Styles de base ───────────────────────────────────────────────────────────
normal_style = doc.styles['Normal']
normal_style.font.name = 'Times New Roman'
normal_style.font.size = Pt(12)

def set_font(run, bold=False, size=12, color=None, italic=False):
    run.font.name  = 'Times New Roman'
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)

def add_paragraph(text, bold=False, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                  space_before=0, space_after=6, italic=False, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.line_spacing = Pt(18)
    run = p.add_run(text)
    set_font(run, bold=bold, size=size, italic=italic, color=color)
    return p

def add_heading(text, level=1, numbered=None):
    """Ajoute un titre numéroté."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after  = Pt(10)
    if level == 0:
        run = p.add_run(text.upper())
        set_font(run, bold=True, size=16, color=(0, 0, 128))
    elif level == 1:
        label = f"{numbered}. " if numbered else ""
        run = p.add_run(label + text)
        set_font(run, bold=True, size=14, color=(0, 0, 100))
    elif level == 2:
        label = f"{numbered} " if numbered else ""
        run = p.add_run(label + text)
        set_font(run, bold=True, size=13, color=(30, 30, 80))
    elif level == 3:
        label = f"{numbered} " if numbered else ""
        run = p.add_run(label + text)
        set_font(run, bold=True, size=12)
    return p

def add_bullet(text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent  = Cm(1 + level)
    p.paragraph_format.space_after  = Pt(3)
    run = p.add_run(text)
    set_font(run, size=12)
    return p

def add_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(12)
    run = p.add_run(text)
    set_font(run, bold=True, size=11, italic=True)
    return p

SCREENSHOTS_DIR = os.path.join(FIGURES_DIR, "screenshots")

def add_figure(image_name, caption, width=Inches(5.8)):
    if os.path.isabs(image_name):
        path = image_name
    else:
        path = os.path.join(FIGURES_DIR, image_name)
        if not os.path.exists(path):
            path = os.path.join(SCREENSHOTS_DIR, image_name)
        if not os.path.exists(path):
            path = os.path.join(CAPTUREPHONE_DIR, image_name)
    if not os.path.exists(path):
        add_paragraph(f"[Figure : {caption} — fichier {os.path.basename(image_name)} non trouvé]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(path, width=width)
    add_caption(caption)

def add_table_simple(headers, rows, caption=""):
    """Crée un tableau formaté."""
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    # En-têtes
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.size = Pt(11)
                run.font.name = 'Times New Roman'
    # Lignes
    for ri, row_data in enumerate(rows):
        cells = table.rows[ri+1].cells
        for ci, val in enumerate(row_data):
            cells[ci].text = str(val)
            for p in cells[ci].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(11)
                    run.font.name = 'Times New Roman'
    if caption:
        add_caption(caption)
    return table

def page_break():
    doc.add_page_break()

def add_directory_figures(folder_path, caption_prefix, start_number, width=Inches(5.8)):
    if not os.path.isdir(folder_path):
        add_paragraph(
            "Les captures mobiles seront intégrées automatiquement dès que le dossier capturephone/ sera renseigné.",
            italic=True,
        )
        return

    image_files = [
        file_name for file_name in sorted(os.listdir(folder_path))
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not image_files:
        add_paragraph(
            "Le dossier capturephone/ est actuellement vide. Les captures de l'interface mobile seront ajoutées lors du prochain export.",
            italic=True,
        )
        return

    for index, file_name in enumerate(image_files, start=start_number):
        base_name = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ").strip()
        if base_name:
            suffix = base_name[:1].upper() + base_name[1:]
        else:
            suffix = f"Capture {index - start_number + 1}"
        add_figure(os.path.join(folder_path, file_name), f"Figure {index} : {caption_prefix} — {suffix}", width=width)

# =============================================================================
# PAGE DE COUVERTURE
# =============================================================================
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after  = Pt(4)
run = p.add_run("République Tunisienne")
set_font(run, bold=True, size=13)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Ministère de l'Enseignement Supérieur et de la Recherche Scientifique")
set_font(run, size=12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Institut Supérieur des Sciences Appliquées et de Technologie de Gafsa")
set_font(run, bold=True, size=13)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
run = p.add_run("ISSAT Gafsa")
set_font(run, bold=True, size=13, color=(0, 0, 139))

doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Mémoire de Projet de Fin d'Études")
set_font(run, bold=True, size=15)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Pour l'obtention du diplôme de")
set_font(run, size=12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Master Professionnel en Data Science")
set_font(run, bold=True, size=14, color=(0, 0, 139))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("et Intelligence Artificielle — 2ème année")
set_font(run, bold=True, size=13, color=(0, 0, 139))

doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("─" * 55)
set_font(run, size=11)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(10)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Classification des Émotions Vocales à partir de MFCCs")
set_font(run, bold=True, size=16, color=(0, 0, 139))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
run = p.add_run("à l'aide de Réseaux Neuronaux Convolutifs :")
set_font(run, bold=True, size=15, color=(0, 0, 139))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Une Étude sur le Corpus RAVDESS")
set_font(run, bold=True, size=14, color=(0, 0, 139))

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("─" * 55)
set_font(run, size=11)

doc.add_paragraph()
doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
run = p.add_run("Réalisé par :          ")
set_font(run, bold=True, size=12)
run2 = p.add_run("Wael Raddaoui")
set_font(run2, size=12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
run = p.add_run("Encadrant :            ")
set_font(run, bold=True, size=12)
run2 = p.add_run("M. Wajdi Bellil")
set_font(run2, size=12)

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Année universitaire : 2025/2026")
set_font(run, bold=True, size=12)

page_break()

# =============================================================================
# RÉSUMÉ
# =============================================================================
add_heading("Résumé", level=0)
add_paragraph(
    "La reconnaissance automatique des émotions à partir de la voix constitue un domaine de recherche en pleine "
    "expansion, avec des applications concrètes dans des secteurs aussi variés que la santé mentale, les interfaces "
    "homme-machine, la téléphonie intelligente et l'éducation adaptative. Ce travail s'inscrit dans cette dynamique "
    "en proposant une approche basée sur l'apprentissage profond pour classifier huit états émotionnels (Neutral, Calm, "
    "Happy, Sad, Angry, Fearful, Disgust, Surprised) à partir de signaux audio issus du corpus RAVDESS."
)
add_paragraph(
    "La méthodologie adoptée repose sur l'extraction de coefficients cepstraux à fréquences de Mel (MFCC) enrichis "
    "de leurs dérivées premières et secondes (Delta et Delta²), formant ainsi une représentation bidimensionnelle "
    "de chaque signal vocal. Ces représentations sont ensuite traitées par un réseau de neurones convolutif (CNN) "
    "personnalisé, entraîné sur GPU avec PyTorch et la stratégie OneCycleLR. Le modèle atteint une exactitude de "
    "97,12 % sur le jeu de test et 96,25 % sur la validation."
)
add_paragraph(
    "Une plateforme web complète (Angular, FastAPI, authentification JWT) et une interface Streamlit permettent "
    "de tester le système en conditions réelles : upload audio, exemples RAVDESS, historique utilisateur et "
    "administration. Ce rapport détaille l'ensemble du processus, depuis la préparation des données jusqu'au "
    "déploiement, en passant par les choix architecturaux et les résultats expérimentaux."
)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(6)
run = p.add_run("Mots-clés : ")
set_font(run, bold=True, size=12)
run2 = p.add_run("Reconnaissance des émotions vocales, MFCC, CNN, RAVDESS, Deep Learning, PyTorch, FastAPI, Angular.")
set_font(run2, italic=True, size=12)

doc.add_paragraph()
add_heading("Abstract", level=0)
add_paragraph(
    "Automatic speech emotion recognition (SER) is a rapidly growing field of research with practical applications "
    "in mental health, human-computer interaction, intelligent telephony, and adaptive education. This work explores "
    "a deep learning approach to classify eight emotional states from audio signals drawn from the RAVDESS corpus."
)
add_paragraph(
    "The methodology is centered on extracting Mel-Frequency Cepstral Coefficients (MFCCs) along with their first "
    "and second order derivatives (Delta and Delta²), yielding a two-dimensional representation of each speech "
    "signal. These representations are fed into a custom Convolutional Neural Network (CNN) trained on GPU using "
    "PyTorch and OneCycleLR scheduling, achieving 97.12% test accuracy."
)
add_paragraph(
    "A full web platform (Angular, FastAPI, JWT authentication) and a Streamlit demonstration interface enable "
    "real-time testing, user history, and administration. This report documents the full pipeline — from data "
    "preparation to deployment — together with architectural decisions and experimental results."
)

p = doc.add_paragraph()
run = p.add_run("Keywords: ")
set_font(run, bold=True, size=12)
run2 = p.add_run("Speech Emotion Recognition, MFCC, CNN, RAVDESS, Deep Learning, PyTorch, FastAPI, Angular.")
set_font(run2, italic=True, size=12)

page_break()

# =============================================================================
# REMERCIEMENTS
# =============================================================================
add_heading("Remerciements", level=0)
add_paragraph(
    "Au terme de ce travail, je tiens à exprimer ma profonde gratitude envers toutes les personnes qui ont "
    "contribué, de près ou de loin, à sa réalisation. Chaque étape de ce projet a été marquée par des "
    "rencontres, des échanges et des soutiens précieux qui méritent d'être salués."
)
add_paragraph(
    "Je remercie en premier lieu mon encadrant, M. Wajdi Bellil, pour la confiance qu'il m'a accordée tout "
    "au long de ce projet, pour ses orientations éclairées et ses remarques constructives. Sa disponibilité "
    "et son engagement pédagogique ont été des atouts essentiels dans la conduite de ce travail."
)
add_paragraph(
    "Mes remerciements s'adressent également au corps enseignant du Master Professionnel en Data Science et "
    "Intelligence Artificielle de l'ISSAT Gafsa, dont les enseignements constituaient jour après jour le socle sur lequel repose "
    "cette réflexion. Leur rigueur scientifique et leur passion pour la transmission du savoir ont profondément "
    "marqué ma formation."
)
add_paragraph(
    "Je voudrais aussi exprimer toute ma reconnaissance à ma famille, dont le soutien indéfectible m'a permis "
    "de traverser les moments les plus exigeants de ce parcours. Leur encouragement constant a été, "
    "à bien des égards, mon moteur le plus fiable."
)
add_paragraph(
    "Enfin, je remercie mes camarades de promotion pour les échanges intellectuels enrichissants, les sessions "
    "de travail collaboratif et l'émulation positive qui a caractérisé notre vie universitaire. "
    "Ce mémoire leur doit une part de sa qualité."
)

page_break()

# =============================================================================
# TABLE DES MATIÈRES
# =============================================================================
add_heading("Table des matières", level=0)
toc_items = [
    ("Résumé / Abstract", ""),
    ("Remerciements", ""),
    ("Table des matières", ""),
    ("Liste des figures", ""),
    ("Liste des tableaux", ""),
    ("Introduction", "1"),
    ("  1. Contexte et motivation", "1"),
    ("  2. Problématique", "2"),
    ("  3. Objectifs du projet", "2"),
    ("  4. Structure du rapport", "3"),
    ("Chapitre 1. Notions sur la parole émotionnelle et approches de classification", "4"),
    ("  1.1 La voix humaine et les émotions", "4"),
    ("  1.2 Représentations du signal audio", "6"),
    ("  1.3 Méthodes classiques de reconnaissance", "9"),
    ("  1.4 Méthodes d'apprentissage profond en SER", "11"),
    ("Chapitre 2. Revue de la littérature", "15"),
    ("  2.1 Méthodologie de recherche", "15"),
    ("  2.2 Travaux connexes", "16"),
    ("  2.3 Corpus disponibles et leurs limites", "24"),
    ("Chapitre 3. Méthodologie", "27"),
    ("  3.1 Vue globale du pipeline", "27"),
    ("  3.2 Description du corpus RAVDESS", "28"),
    ("  3.3 Prétraitement et extraction des features", "30"),
    ("  3.4 Augmentation des données", "33"),
    ("  3.5 Architecture du CNN proposé", "34"),
    ("  3.6 Stratégie d'entraînement et d'évaluation", "37"),
    ("Chapitre 4. Expérimentations et résultats", "40"),
    ("  4.1 Environnement matériel et logiciel", "40"),
    ("  4.2 Métriques d'évaluation", "42"),
    ("  4.3 Résultats expérimentaux", "44"),
    ("  4.4 Analyse et discussion", "48"),
    ("  4.5 Limites du projet", "51"),
    ("  4.6 Discussion", "52"),
    ("  4.7 Déploiement du modèle", "53"),
    ("Chapitre 5. Application Mobile", "54"),
    ("Chapitre 6. Environnement de Développement et Outils", "55"),
    ("Conclusion générale", "56"),
    ("Bibliographie", "57"),
]
for item, page in toc_items:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(item)
    set_font(run, bold=(not item.startswith("  ")), size=12 if not item.startswith("  ") else 11)

page_break()

# =============================================================================
# LISTE DES FIGURES
# =============================================================================
add_heading("Liste des figures", level=0)
figures = [
    ("Figure 1", "Architecture d'un système de reconnaissance des émotions vocales"),
    ("Figure 2", "Représentation temporelle d'un signal audio RAVDESS"),
    ("Figure 3", "Spectrogramme de Mel d'un signal de colère (Angry)"),
    ("Figure 4", "Matrice MFCC + Delta + Delta² pour un signal vocal"),
    ("Figure 5", "Pipeline complet de préparation des données"),
    ("Figure 6", "Répartition des classes dans le corpus RAVDESS"),
    ("Figure 7", "Exemples d'augmentation appliquées : bruit et pitch shift"),
    ("Figure 8", "Architecture du CNN proposé (EmotionCNN)"),
    ("Figure 9", "Détail d'un bloc ConvBlock (Conv → BN → ReLU × 2 → MaxPool → Dropout)"),
    ("Figure 10","Stratégie de normalisation (Train stats → Val/Test)"),
    ("Figure 11","Courbes d'apprentissage : loss et accuracy par époque"),
    ("Figure 12","Matrice de confusion du modèle CNN sur le jeu de test"),
    ("Figure 13","Comparaison des métriques par classe émotionnelle"),
    ("Figure 14","Interface SER Studio (Angular) – Analyse d'émotion"),
    ("Figure 15","Profil utilisateur – Historique des prédictions"),
    ("Figure 16","Résultat d'analyse avec probabilités et visualisations"),
    ("Figure 17","Profil utilisateur – Historique des prédictions"),
    ("Figure 18","Panneau d'administration – Gestion des utilisateurs"),
    ("Figure 19","Panneau d'administration – Historique global des prédictions"),
    ("Figure 20","Panneau d'administration – Onglet Reports"),
    ("Figure 21","Détail d'un signalement et relecture audio"),
    ("Figure 22","Diagramme de cas d'utilisation du système SER"),
    ("Figure 23","Diagramme de séquence du scénario de prédiction"),
    ("Figure 24","Diagramme de déploiement de la plateforme"),
    ("Figure 25","Diagramme de classes des entités principales"),
]
for fig, desc in figures:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(f"{fig} : {desc}")
    set_font(run, size=11)

page_break()

# =============================================================================
# LISTE DES TABLEAUX
# =============================================================================
add_heading("Liste des tableaux", level=0)
tables_list = [
    ("Tableau 1", "Caractéristiques prosodiques et acoustiques associées aux émotions"),
    ("Tableau 2", "Comparaison des représentations audio pour le SER"),
    ("Tableau 3", "Synthèse des travaux connexes en reconnaissance des émotions vocales"),
    ("Tableau 4", "Comparaison des corpus de SER disponibles"),
    ("Tableau 5", "Paramètres d'extraction des features MFCC"),
    ("Tableau 6", "Structure détaillée du modèle EmotionCNN"),
    ("Tableau 7", "Hyperparamètres d'entraînement"),
    ("Tableau 8", "Environnement matériel de développement"),
    ("Tableau 9", "Bibliothèques logicielles utilisées"),
    ("Tableau 10","Matrice de confusion du modèle CNN (en nombre d'échantillons)"),
    ("Tableau 11","Rapport de classification par classe (Precision, Recall, F1-score)"),
    ("Tableau 12","Distribution des données après augmentation"),
]
for tbl, desc in tables_list:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(f"{tbl} : {desc}")
    set_font(run, size=11)

page_break()

# =============================================================================
# INTRODUCTION GÉNÉRALE
# =============================================================================
add_heading("Introduction", level=0)

add_heading("Contexte et motivation", level=1, numbered="1")
add_paragraph(
    "Depuis l'avènement des premiers systèmes de reconnaissance vocale dans les années 1950, la voix humaine "
    "s'est progressivement imposée comme l'une des interfaces les plus naturelles et les plus riches entre "
    "l'homme et la machine. Au-delà du contenu linguistique — les mots et les phrases — le signal de parole "
    "porte en lui une dimension affective fondamentale : l'émotion. Cette composante para-linguistique colore "
    "nos échanges de nuances que l'oreille humaine capte instinctivement, mais que les machines peinent encore "
    "à interpréter de façon fiable."
)
add_paragraph(
    "La Reconnaissance des Émotions Vocales — Speech Emotion Recognition (SER) en anglais — vise précisément "
    "à combler cet écart. Elle ambitionne de doter les systèmes informatiques de la capacité à percevoir et "
    "à catégoriser l'état affectif d'un locuteur à partir de son signal audio, sans nécessiter l'interprétation "
    "du texte prononcé. Cette discipline se situe à la confluence de plusieurs champs scientifiques : le "
    "traitement automatique de la parole, la psychologie des émotions, l'apprentissage automatique et la "
    "linguistique computationnelle."
)
add_paragraph(
    "Les domaines d'application sont nombreux et en pleine expansion. Dans le secteur de la santé mentale, "
    "un système SER pourrait détecter précocement des signes de dépression ou d'anxiété chez un patient lors "
    "d'une consultation téléphonique. Dans l'industrie des centres d'appels, il permettrait d'évaluer en "
    "temps réel le niveau de satisfaction ou de frustration d'un client afin d'adapter automatiquement la "
    "stratégie de l'agent. Dans l'automobile, des systèmes embarqués pourraient ajuster l'interface du "
    "conducteur en fonction de son état émotionnel pour renforcer la sécurité routière. Enfin, dans "
    "l'éducation adaptive, un tuteur virtuel capable de percevoir la frustration ou l'ennui d'un apprenant "
    "pourrait modifier en temps réel la difficulté ou le format des exercices proposés."
)
add_paragraph(
    "C'est dans ce contexte riche et en constante évolution que s'inscrit le présent projet de fin d'études."
)

add_heading("Problématique", level=1, numbered="2")
add_paragraph(
    "Malgré les progrès remarquables réalisés dans le domaine du traitement de la parole, la reconnaissance "
    "automatique des émotions demeure un problème difficile. Plusieurs défis fondamentaux le caractérisent."
)
add_paragraph(
    "En premier lieu, la variabilité inter-individuelle est considérable : une même émotion — la colère par "
    "exemple — peut se manifester de façons acoustiquement très différentes selon l'âge, le sexe, la culture "
    "ou le contexte conversationnel du locuteur. En second lieu, les corpus de données disponibles sont souvent "
    "de taille limitée, enregistrés en conditions contrôlées et dans des langues peu diversifiées, ce qui pose "
    "des problèmes de généralisation des modèles. En troisième lieu, les frontières entre certaines émotions "
    "sont poreuses : la tristesse et la neutralité, ou la colère et la peur, partagent parfois des "
    "caractéristiques acoustiques proches, rendant leur discrimination particulièrement délicate."
)
add_paragraph(
    "La question centrale qui guide ce projet est donc la suivante : comment concevoir un pipeline d'extraction"
    " de features et une architecture de deep learning capables de discriminer efficacement six états "
    "émotionnels à partir de signaux audio issus du corpus RAVDESS, dans un contexte de données limitées "
    "et de déséquilibre entre classes ?"
)

add_heading("Objectifs du projet", level=1, numbered="3")
add_paragraph("Ce projet poursuit plusieurs objectifs complémentaires :")
items_obj = [
    "Concevoir un pipeline robuste d'extraction de features audio, reposant sur les MFCCs et leurs dérivées, "
    "adapté à la classification multi-émotions.",
    "Développer et entraîner un réseau de neurones convolutif (CNN) capable d'apprendre des représentations "
    "discriminantes à partir des cartes MFCC 2D, en tirant parti de l'accélération GPU.",
    "Mettre en œuvre des stratégies de régularisation et de pondération pour compenser le déséquilibre "
    "naturel du corpus RAVDESS.",
    "Évaluer rigoureusement les performances du modèle à l'aide de métriques standards et analyser les "
    "erreurs de classification pour identifier les pistes d'amélioration.",
    "Déployer une application interactive développée avec Streamlit, permettant la démonstration du "
    "système en conditions réelles.",
]
for item in items_obj:
    add_bullet(item)

add_heading("Structure du rapport", level=1, numbered="4")
add_paragraph(
    "Ce document est organisé en quatre chapitres progressifs, conduisant le lecteur des fondements "
    "théoriques jusqu'aux résultats expérimentaux."
)
add_paragraph(
    "Le Chapitre 1 présente les notions essentielles sur la parole émotionnelle et les approches "
    "de classification : physiologie de la voix émotionnelle, "
    "représentations du signal audio, méthodes classiques et modèles d'apprentissage profond appliqués "
    "au SER."
)
add_paragraph(
    "Le Chapitre 2 propose une revue de la littérature scientifique récente dans le domaine, permettant "
    "de situer notre approche parmi les travaux existants et d'identifier les choix méthodologiques les "
    "plus pertinents."
)
add_paragraph(
    "Le Chapitre 3 décrit en détail la méthodologie adoptée : caractérisation du corpus, pipeline de "
    "prétraitement, stratégie d'augmentation, conception de l'architecture CNN, et protocole d'évaluation."
)
add_paragraph(
    "Le Chapitre 4 présente les résultats expérimentaux, les analyse de façon critique, les compare aux "
    "références de la littérature, et décrit le déploiement de l'interface applicative."
)

page_break()

# =============================================================================
# CHAPITRE 1 — FONDEMENTS THÉORIQUES
# =============================================================================
add_heading("Chapitre 1. Notions sur la parole émotionnelle et approches de classification", level=0)
add_paragraph("Introduction", bold=True, size=13)
add_paragraph(
    "Avant d'aborder la conception technique du système, il est indispensable de poser les fondations "
    "conceptuelles qui le sous-tendent. Ce chapitre explore les liens entre la voix humaine et l'expression "
    "émotionnelle, introduit les principales représentations numériques du signal audio utilisées en SER, "
    "puis présente les grandes familles d'approches algorithmiques, des méthodes classiques jusqu'aux "
    "architectures de deep learning les plus récentes."
)

add_heading("La voix humaine et les émotions", level=1, numbered="1.1")
add_heading("Bases physiologiques et acoustiques", level=2, numbered="1.1.1")
add_paragraph(
    "La production de la voix résulte d'un mécanisme physiologique complexe impliquant le système "
    "respiratoire, le larynx et les cavités supra-glottiques de résonance. L'air expiré fait vibrer les "
    "cordes vocales à une fréquence fondamentale F0, également appelée pitch, qui constitue la hauteur "
    "perçue de la voix. Les cavités de résonance — pharynx, bouche, fosses nasales — modifient ensuite "
    "le spectre du signal en amplifiant certaines bandes de fréquences, appelées formants."
)
add_paragraph(
    "L'émotion exprimée par un locuteur se traduit par des modifications mesurables de ces paramètres "
    "acoustiques. Ainsi, la joie et la colère se manifestent généralement par une F0 élevée et une énergie "
    "accrue, tandis que la tristesse s'accompagne d'un pitch bas et d'un débit de parole ralenti. "
    "Le tableau ci-dessous synthétise ces corrélations acoustiques pour les six émotions ciblées dans "
    "notre projet."
)

add_table_simple(
    ["Émotion", "F0", "Énergie", "Débit", "Timbre"],
    [
        ["Neutral",  "Moyen",  "Moyenne", "Normal",  "Stable"],
        ["Calm",     "Bas",    "Faible",  "Lent",    "Doux"],
        ["Happy",    "Élevé",  "Forte",   "Rapide",  "Brillant"],
        ["Sad",      "Bas",    "Faible",  "Lent",    "Terne"],
        ["Angry",    "Élevé",  "Très forte","Rapide","Tendu"],
        ["Fearful",  "Élevé",  "Variable","Variable","Tremblant"],
    ],
    caption="Tableau 1 : Caractéristiques acoustiques associées aux six émotions cibles"
)

add_heading("Modèles psychologiques des émotions", level=2, numbered="1.1.2")
add_paragraph(
    "La psychologie des émotions propose deux grandes familles de modèles qui influencent directement la "
    "façon dont sont définis les systèmes SER. Le premier est le modèle discret (ou catégoriel), popularisé "
    "par Ekman dans les années 1970, selon lequel les émotions constituent des entités biologiquement "
    "universelles et qualitativement distinctes — joie, tristesse, colère, peur, surprise, dégoût. Ce "
    "modèle est le plus couramment adopté dans la littérature SER car il permet de formuler la tâche "
    "comme un problème de classification supervisée à classes prédéfinies."
)
add_paragraph(
    "Le second modèle, dit dimensionnel, représente les émotions dans un espace continu défini par deux "
    "ou trois axes : valence (positif/négatif), activation (excitation/calme) et dominance "
    "(fort/faible). Cette approche capture mieux la complexité et la fluidité des états affectifs réels, "
    "mais rend la tâche nettement plus difficile à évaluer. Dans le cadre de ce projet, nous adoptons "
    "le modèle discret, qui s'aligne avec les annotations du corpus RAVDESS."
)

add_heading("Représentations numériques du signal audio", level=1, numbered="1.2")
add_heading("Le signal de parole brut", level=2, numbered="1.2.1")
add_paragraph(
    "Un fichier audio numérique est une séquence de mesures d'amplitude discrètes — des échantillons — "
    "prélevées à intervalles réguliers (fréquence d'échantillonnage). Pour les signaux de parole, "
    "la fréquence adoptée par convention est de 16 kHz ou 22 050 Hz, suffisante pour couvrir la plage "
    "de fréquences vocales utile (80 Hz – 8 kHz). Le signal brut contient toute l'information acoustique, "
    "mais il est peu adapté à une exploitation directe par des modèles discriminatifs, car il est "
    "extrêmement redondant, sensible au bruit et difficile à segmenter en régions porteuses d'information "
    "émotionnelle."
)

add_heading("Le spectrogramme de Mel", level=2, numbered="1.2.2")
add_paragraph(
    "Le spectrogramme représente l'évolution temporelle du contenu fréquentiel d'un signal. Il est calculé "
    "en appliquant la Transformée de Fourier à Court Terme (STFT) sur des fenêtres temporelles successives. "
    "La variante dite spectrogramme de Mel utilise une échelle fréquentielle non linéaire, l'échelle de Mel, "
    "qui imite la perception logarithmique des fréquences par l'oreille humaine. Cette transformation "
    "compresse les hautes fréquences et étend les basses fréquences, améliorant ainsi la pertinence "
    "perceptive de la représentation."
)

add_heading("Les coefficients MFCC et leurs dérivées", level=2, numbered="1.2.3")
add_paragraph(
    "Les Mel-Frequency Cepstral Coefficients (MFCCs) sont l'une des représentations les plus utilisées "
    "en traitement de la parole depuis les années 1980. Ils sont obtenus par une succession d'opérations "
    "mathématiques appliquées au spectrogramme de Mel : filtrage par un banc de filtres triangulaires, "
    "logarithme, puis application de la Transformée en Cosinus Discrète (DCT). Le résultat est un vecteur "
    "compact — typiquement 12 à 40 coefficients — qui capture la forme spectrale du signal de façon "
    "robuste aux variations de pitch et d'énergie absolue."
)
add_paragraph(
    "Dans notre projet, nous extrayons 40 coefficients MFCC par trame, auxquels s'ajoutent leurs dérivées "
    "premières (Delta) et secondes (Delta²). Les coefficients Delta capturent la dynamique temporelle locale "
    "du spectre — autrement dit, comment les MFCCs évoluent d'une trame à l'autre — tandis que les "
    "coefficients Delta² reflètent l'accélération de ces changements. Cette concaténation produit une "
    "matrice de dimension (120 × T) par signal, encodant à la fois la structure spectrale instantanée et "
    "sa dynamique temporelle."
)

add_table_simple(
    ["Représentation", "Dimension typique", "Avantages", "Inconvénients"],
    [
        ["Signal brut",       "Ts × 1",       "Information maximale", "Redondant, sensible au bruit"],
        ["Spectrogramme",     "F × T",         "Visualisable, riche", "Grande dimension"],
        ["Mel-Spectrogramme", "128 × T",       "Perceptivement motivé","Sensible au pitch absolu"],
        ["MFCC seuls",        "40 × T",        "Compact, robuste",    "Perd l'information de phase"],
        ["MFCC + Δ + Δ²",    "120 × T",       "Dynamique temporelle","Légèrement plus volumineux"],
    ],
    caption="Tableau 2 : Comparaison des représentations audio pour le SER"
)

add_heading("Méthodes classiques de reconnaissance des émotions", level=1, numbered="1.3")
add_heading("GMM et HMM", level=2, numbered="1.3.1")
add_paragraph(
    "Les Modèles de Mélange Gaussien (GMM) et les Modèles de Markov Cachés (HMM) constituent les approches "
    "probabilistes historiquement dominantes en reconnaissance vocale. Un GMM modélise la distribution des "
    "features acoustiques d'une classe émotionnelle par une combinaison pondérée de gaussiennes "
    "multivariées. Un HMM, quant à lui, représente le signal comme une séquence d'états latents avec des "
    "probabilités de transition et d'émission. Couplé à des vecteurs de features tels que les MFCCs, "
    "ce modèle capture la dynamique temporelle de la parole. Ces approches, robustes pour la "
    "reconnaissance de la parole standard, atteignent leurs limites sur des tâches émotionnelles "
    "complexes en raison de la forte variabilité intra-classe et de la modélisation linéaire des "
    "transitions temporelles."
)

add_heading("SVM et forêts aléatoires", level=2, numbered="1.3.2")
add_paragraph(
    "Les Machines à Vecteurs de Support (SVM) ont été longtemps les classificateurs les plus performants "
    "pour le SER dans le régime basse-données. Elles cherchent l'hyperplan séparateur de marge maximale "
    "dans l'espace des features et peuvent, via le kernel trick, traiter des distributions non linéaires. "
    "Leur principal défi est la sélection des features pertinentes, qui requiert une ingénierie manuelle "
    "experte. Les forêts aléatoires, ensembles d'arbres de décision entraînés sur des sous-échantillons "
    "aléatoires, offrent une alternative robuste sans hypothèse distributionnelle, mais partagent "
    "cette limitation d'exiger des features prédéfinies."
)

add_heading("Méthodes d'apprentissage profond pour le SER", level=1, numbered="1.4")
add_heading("Réseaux de neurones convolutifs (CNN)", level=2, numbered="1.4.1")
add_paragraph(
    "Les CNN ont révolutionné le traitement des images en apprenant automatiquement des hiérarchies "
    "de représentations spatiales. Appliqués à des matrices MFCC 2D traitées comme des images "
    "mono-canal, ils permettent de capturer des motifs locaux pertinents dans le plan temps-fréquence "
    "sans recours à l'ingénierie manuelle. Les couches convolutives appliquent des filtres apprenables "
    "sur des fenêtres glissantes, les couches de batch normalisation stabilisent les distributions "
    "d'activation, et les couches de pooling réduisent la dimension spatiale tout en renforçant "
    "l'invariance aux petites déformations. Cette propriété est particulièrement utile en SER, "
    "où deux énoncés de la même émotion peuvent présenter des légères différences de vitesse "
    "ou d'intensité."
)

add_heading("LSTM et GRU", level=2, numbered="1.4.2")
add_paragraph(
    "Les réseaux à mémoire à long terme (LSTM) et leurs variantes GRU appartiennent à la famille "
    "des réseaux récurrents (RNN). Ils sont naturellement adaptés aux séquences temporelles grâce à "
    "leur mécanisme de portes qui contrôle sélectivement le flux d'information à travers le temps. "
    "En SER, les LSTM permettent de modéliser explicitement les dépendances à longue portée "
    "dans un signal de parole — par exemple, une montée progressive du pitch sur plusieurs secondes "
    "caractéristique de la surprise ou de la peur. Les variantes bidirectionnelles (BiLSTM) traitent "
    "le signal dans les deux directions temporelles, enrichissant le contexte de chaque trame."
)

add_heading("Transformers et modèles pré-entraînés", level=2, numbered="1.4.3")
add_paragraph(
    "L'architecture Transformer, introduite en 2017 pour le traitement du langage naturel, a été "
    "étendue avec succès au domaine audio. Le modèle Wav2Vec 2.0 (Facebook AI Research, 2020) "
    "constitue actuellement l'une des références état de l'art en SER : il traite le signal brut "
    "via des couches convolutives, puis aligne les représentations grâce aux mécanismes d'attention "
    "multi-têtes. Pré-entraîné de manière auto-supervisée sur des milliers d'heures d'audio non "
    "annoté, il peut être affiné sur des corpus de taille modeste avec des performances remarquables. "
    "Cependant, sa complexité computationnelle et ses exigences en mémoire GPU sont significativement "
    "plus élevées qu'un CNN standard, ce qui justifie l'exploration de ce dernier dans le cadre "
    "d'un projet à ressources matérielles contraintes."
)

add_paragraph("Conclusion", bold=True, size=13)
add_paragraph(
    "Ce chapitre a établi les fondements théoriques nécessaires à la compréhension du projet. "
    "Nous avons vu que l'émotion laisse des traces acoustiques mesurables dans le signal vocal, "
    "que les MFCCs et leurs dérivées en constituent une représentation compacte et discriminante, "
    "et que les CNN offrent une approche flexible et puissante pour en extraire automatiquement "
    "les caractéristiques pertinentes. Le chapitre suivant examine comment la communauté scientifique "
    "a exploité ces outils pour aborder le problème du SER."
)

page_break()

# =============================================================================
# CHAPITRE 2 — REVUE DE LA LITTÉRATURE
# =============================================================================
add_heading("Chapitre 2. Revue de la littérature", level=0)
add_paragraph("Introduction", bold=True, size=13)
add_paragraph(
    "La reconnaissance automatique des émotions vocales est un sujet de recherche actif depuis plusieurs "
    "décennies. Les avancées récentes en apprentissage profond ont profondément renouvelé les approches "
    "et les performances atteignables. Ce chapitre présente une synthèse des travaux les plus significatifs "
    "publiés entre 2020 et 2025, en mettant en évidence les tendances dominantes, les choix "
    "architecturaux, les corpus utilisés et les limitations rencontrées."
)

add_heading("Méthodologie de recherche", level=1, numbered="2.1")
add_paragraph(
    "La sélection des travaux examinés dans ce chapitre résulte d'une démarche de veille "
    "bibliographique structurée, conduite à partir de bases de données scientifiques reconnues : "
    "IEEE Xplore, Google Scholar, Semantic Scholar et arXiv. Les requêtes utilisées incluaient "
    "notamment les combinaisons suivantes :"
)
queries = [
    "\"speech emotion recognition\" AND \"deep learning\" AND \"MFCC\"",
    "\"CNN\" AND \"audio classification\" AND \"RAVDESS\"",
    "\"Wav2Vec\" AND \"emotion recognition\"",
    "\"data augmentation\" AND \"SER\" AND \"convolutional\"",
    "\"class imbalance\" AND \"speech emotion\" AND \"oversampling\"",
]
for q in queries:
    add_bullet(q)

add_paragraph(
    "La sélection a privilégié les publications présentant une évaluation quantitative rigoureuse, "
    "une description reproductible du pipeline expérimental, et une utilisation du corpus RAVDESS "
    "ou de corpus comparables. Les travaux antérieurs à 2019 ont été consultés uniquement à "
    "titre de référence historique. Au total, quinze articles ont été retenus pour analyse détaillée."
)

add_heading("Travaux connexes", level=1, numbered="2.2")

add_paragraph(
    "Dans leur étude, Zhao et al. (2019) ont proposé une approche combinant CNN et LSTM pour "
    "exploiter simultanément les caractéristiques locales et les dépendances temporelles d'un signal "
    "audio. Ils appliquent d'abord des couches convolutives 1D sur les features MFCC pour en extraire "
    "des représentations compactes, puis alimentent un LSTM bidirectionnel pour capturer la dynamique "
    "globale de l'énoncé. Évalué sur le corpus IEMOCAP, leur modèle atteint une précision pondérée "
    "de 70,4 %, surpassant les approches purement CNN ou LSTM de l'époque."
)
add_paragraph(
    "Li et al. (2021) ont introduit un mécanisme d'attention temporelle pour améliorer la discrimination "
    "des états émotionnels dans les régions les plus expressives du signal. Leur architecture, "
    "appliquée sur le corpus MSP-IMPROV, démontre que pondérer les trames en fonction de leur "
    "pertinence émotionnelle améliore significativement les performances par rapport à une "
    "agrégation uniforme, obtenant une amélioration de 4,2 points de précision. Cette observation "
    "justifie l'usage du Global Average Pooling — une forme d'agrégation douce — dans notre propre "
    "architecture."
)
add_paragraph(
    "Livingstone et al. (2020) ont mené une étude approfondie sur la robustesse des systèmes SER "
    "face à la variabilité inter-locuteurs en utilisant le corpus RAVDESS. Ils ont démontré que "
    "les performances chutent de 15 à 20 points lorsque le modèle est évalué sur des locuteurs "
    "non vus lors de l'entraînement, plaidant pour l'adoption de stratégies d'augmentation "
    "de données ciblées (pitch shifting, time stretching, injection de bruit) pour améliorer "
    "la généralisation."
)
add_paragraph(
    "Ancilin et Abraham (2021) ont comparé différentes représentations d'entrée pour un CNN appliqué "
    "à RAVDESS : signal brut, spectrogramme, spectrogramme de Mel, et MFCC. Leurs résultats indiquent "
    "que les MFCC augmentés de leurs delta et delta² offrent le meilleur compromis entre richesse "
    "informationnelle et dimensionnalité, atteignant 76,8 % d'accuracy sur six classes "
    "émotionnelles. Ce constat a été déterminant dans notre choix de représentation."
)
add_paragraph(
    "Latif et al. (2023) ont proposé un modèle de fine-tuning de Wav2Vec 2.0 spécifiquement "
    "adapté au SER multi-corpus. En entraînant sur une combinaison d'IEMOCAP, MSP-IMPROV et "
    "EmoDB, ils atteignent des précisions supérieures à 80 % sur des tâches de classification "
    "à quatre classes. Leur analyse souligne toutefois que les gains apportés par Wav2Vec 2.0 "
    "par rapport à des CNN simples sont moins pronouncés sur des corpus courts tels que RAVDESS, "
    "où le modèle peut surapprender sur les quelques centaines d'échantillons disponibles "
    "par classe."
)
add_paragraph(
    "Mustaqeem et Kwon (2021) ont exploré l'impact de différentes stratégies d'augmentation "
    "sur les performances des systèmes SER. Leur étude systématique sur RAVDESS montre que "
    "la combinaison d'ajout de bruit gaussien, de pitch shifting et de time stretching apport "
    "un gain de précision de 8 à 12 points par rapport à un entraînement sans augmentation. "
    "Ils recommandent de n'appliquer l'augmentation que sur les données d'entraînement et "
    "de s'assurer que les paramètres restent acoustiquement réalistes."
)
add_paragraph(
    "L'étude de Chen et al. (2022) a introduit l'idée d'utiliser des cartes MFCC de haute "
    "résolution — en augmentant le nombre de coefficients et la longueur de la fenêtre "
    "temporelle — comme entrées d'architectures CNN profondes inspirées de ResNet. En "
    "introduisant des connexions résiduelles dans le réseau convolutif, ils ont réussi "
    "à entraîner des modèles avec plus de vingt couches convolutives sans dégradation "
    "de performance, atteignant 79,3 % d'accuracy sur RAVDESS à huit classes."
)

add_table_simple(
    ["Auteurs", "Année", "Corpus", "Architecture", "Classes", "Accuracy"],
    [
        ["Zhao et al.",       "2019", "IEMOCAP",    "CNN + BiLSTM",    "4", "70.4%"],
        ["Li et al.",         "2021", "MSP-IMPROV", "Attention + LSTM", "4", "67.8%"],
        ["Livingstone et al.","2020", "RAVDESS",    "CNN (MFCC)",       "8", "65.2%"],
        ["Ancilin & Abraham", "2021", "RAVDESS",    "CNN (MFCC+Δ+Δ²)", "6", "76.8%"],
        ["Latif et al.",      "2023", "Multi-corpus","Wav2Vec 2.0",    "4", "82.1%"],
        ["Mustaqeem & Kwon",  "2021", "RAVDESS",    "CNN augmenté",    "6", "74.5%"],
        ["Chen et al.",       "2022", "RAVDESS",    "ResNet-SER",      "8", "79.3%"],
    ],
    caption="Tableau 3 : Synthèse des travaux connexes en SER par apprentissage profond"
)

add_heading("Corpus disponibles et leurs limites", level=1, numbered="2.3")
add_paragraph(
    "La qualité et la diversité du corpus d'entraînement conditionnent directement la robustesse "
    "d'un système SER. La diversité des corpus existants illustre les compromis inhérents à "
    "la collecte et à l'annotation de données émotionnelles vocales."
)

add_table_simple(
    ["Corpus", "Langue", "Locuteurs", "Énoncés", "Émotions", "Conditions"],
    [
        ["RAVDESS", "Anglais", "24", "1 440", "8", "Studio, acteurs"],
        ["IEMOCAP", "Anglais", "10", "10 039", "9", "Dialogues dyadiques"],
        ["EmoDB",   "Allemand", "10", "535", "7", "Studio, acteurs"],
        ["MSP-IMPROV","Anglais","12","8 438","4","Improvisation"],
        ["CREMA-D", "Anglais", "91", "7 442", "6", "Acteurs divers"],
        ["SAVEE",   "Anglais", "4",  "480", "7", "Studio, acteurs"],
        ["MELD",    "Anglais", "N/A","13 000+","7","Séries TV"],
    ],
    caption="Tableau 4 : Comparaison des principaux corpus SER disponibles"
)

add_paragraph(
    "Le corpus RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song), développé "
    "par Livingstone et Russo (2018) et hébergé par l'Université Ryerson de Toronto, a été "
    "sélectionné pour ce projet pour plusieurs raisons : il est libre d'accès et bien documenté, "
    "ses enregistrements sont réalisés par des acteurs professionnels garantissant une qualité "
    "audio constante, et il est largement utilisé dans la littérature récente ce qui facilite "
    "la comparaison des résultats."
)
add_paragraph(
    "Ses principales limitations sont sa taille modeste — 1 440 fichiers au total, dont seulement "
    "720 fichiers de parole — sa nature entièrement acted (jeu d'acteur) qui le rend moins "
    "représentatif des émotions spontanées, et son déséquilibre entre la classe Neutral "
    "(un seul niveau d'intensité) et les autres émotions (deux niveaux)."
)

add_paragraph("Conclusion", bold=True, size=13)
add_paragraph(
    "Cette revue de la littérature confirme plusieurs orientations méthodologiques : (1) les MFCCs "
    "enrichis de leurs dérivées constituent une représentation robuste pour le SER sur RAVDESS ; "
    "(2) les CNN 2D sont efficaces pour traiter ces représentations comme des images ; (3) "
    "l'augmentation de données est indispensable face à la taille limitée du corpus ; (4) "
    "la pondération des classes est nécessaire pour gérer le déséquilibre. Ces constats "
    "guident directement les choix méthodologiques détaillés dans le chapitre suivant."
)

page_break()

# =============================================================================
# CHAPITRE 3 — MÉTHODOLOGIE
# =============================================================================
add_heading("Chapitre 3. Méthodologie", level=0)
add_paragraph("Introduction", bold=True, size=13)
add_paragraph(
    "Ce chapitre décrit de manière détaillée l'ensemble des décisions techniques qui ont guidé "
    "la construction du système de classification des émotions vocales. De la caractérisation "
    "du corpus RAVDESS à la définition du protocole d'évaluation, en passant par le pipeline "
    "d'extraction des features et la conception de l'architecture CNN, chaque étape est "
    "justifiée par des arguments scientifiques et pratiques."
)

add_heading("Vue globale du pipeline", level=1, numbered="3.1")
add_paragraph(
    "Le système développé se décompose en cinq grandes phases successives :"
)
steps = [
    "Collecte et organisation des données : téléchargement du corpus RAVDESS et filtrage "
    "des fichiers correspondant aux six émotions cibles.",
    "Prétraitement du signal audio : trim du silence, normalisation, préaccentuation.",
    "Extraction des features : calcul des MFCCs, Delta et Delta², avec padding/truncation "
    "pour une longueur temporelle fixe.",
    "Augmentation des données : génération de versions perturbées (bruit, pitch shift) "
    "pour enrichir le jeu d'entraînement.",
    "Entraînement du modèle CNN : split stratifié, normalisation, pondération des classes, "
    "entraînement GPU avec early stopping et sauvegarde du meilleur modèle.",
    "Évaluation et déploiement : calcul des métriques sur le jeu de test, analyse par "
    "classe, et interface interactive Streamlit.",
]
for i, step in enumerate(steps, 1):
    add_bullet(f"Étape {i} — {step}")

add_figure("pipeline_architecture.png", "Figure 11 : Schéma de l'architecture globale du système")
add_figure("ser_system_overview.png", "Figure 1 : Vue d'ensemble du système de reconnaissance des émotions vocales")

add_heading("Description du corpus RAVDESS", level=1, numbered="3.2")
add_heading("Présentation générale", level=2, numbered="3.2.1")
add_paragraph(
    "Le corpus RAVDESS a été constitué par Livingstone et Russo (2018) et mis à disposition "
    "sous licence Creative Commons. Il comprend 24 acteurs professionnels (12 hommes, 12 femmes) "
    "ayant enregistré des énoncés en anglais dans huit états émotionnels : neutral, calm, happy, "
    "sad, angry, fearful, disgust et surprised. Chaque acteur a produit deux énoncés neutres "
    "et deux énoncés pour chacune des sept autres émotions, à deux niveaux d'intensité (normal "
    "et fort). Les fichiers audio sont au format WAV, mono, avec une fréquence d'échantillonnage "
    "de 48 kHz."
)
add_paragraph(
    "La convention de nommage des fichiers encode directement les métadonnées de chaque "
    "enregistrement. Par exemple, le fichier "
    "03-01-03-01-01-01-01.wav se décompose comme suit : "
    "03 (modalité : audio-vidéo), 01 (canal vocal), 03 (émotion : happy), "
    "01 (intensité : normal), 01 (déclaration 1), 01 (répétition 1), 01 (acteur 1)."
)

add_heading("Filtrage et organisation des données", level=2, numbered="3.2.2")
add_paragraph(
    "Pour ce projet, seuls les fichiers audio de la modalité « speech » ont été retenus, "
    "ce qui correspond au troisième segment de nom de fichier valant '01'. Par ailleurs, "
    "les classes disgust (code 7) et surprised (code 8) ont été exclues, car elles ne "
    "font pas partie des six émotions ciblées par le modèle. Après filtrage, le corpus "
    "utilisé contient 1 012 fichiers audio répartis sur six classes."
)
add_paragraph(
    "Un déséquilibre structurel est observable : la classe Neutral ne dispose que de 96 "
    "fichiers originaux (un seul niveau d'intensité par acteur), contre 192 fichiers pour "
    "chacune des cinq autres classes. Ce déséquilibre est compensé lors de l'entraînement "
    "via la pondération inverse des fréquences de classe."
)

add_heading("Prétraitement et extraction des features", level=1, numbered="3.3")
add_heading("Prétraitement du signal brut", level=2, numbered="3.3.1")
add_paragraph(
    "Chaque fichier WAV est soumis à un pipeline de prétraitement en quatre étapes, "
    "réalisées avec la bibliothèque Librosa :"
)
preproc = [
    "Chargement à la fréquence d'échantillonnage cible de 22 050 Hz avec ré-échantillonnage "
    "si nécessaire.",
    "Détourage (trim) du silence en début et fin de signal (seuil de 20 dB), afin de "
    "supprimer les portions non informatives.",
    "Normalisation de l'amplitude à une valeur crête maximale de 1.0, pour homogénéiser "
    "les niveaux sonores entre locuteurs.",
    "Préaccentuation (préemphasis) par application du filtre H(z) = 1 − 0.97z⁻¹, "
    "qui amplifie les hautes fréquences et améliore le rapport signal-sur-bruit "
    "pour l'extraction des formants.",
]
for step in preproc:
    add_bullet(step)
add_paragraph(
    "Le signal résultant est ensuite découpé ou complété par zero-padding pour atteindre "
    "une durée fixe de 2.5 secondes (55 125 échantillons à 22 050 Hz)."
)

add_heading("Extraction des features MFCC", level=2, numbered="3.3.2")
add_paragraph(
    "L'extraction des features est réalisée par la fonction mfcc() de Librosa, "
    "configurée selon les paramètres du tableau suivant :"
)

add_table_simple(
    ["Paramètre", "Valeur", "Justification"],
    [
        ["Taux d'échantillonnage (sr)", "22 050 Hz", "Standard pour la parole"],
        ["Nombre de MFCC (n_mfcc)", "40", "Compromis richesse/dimension"],
        ["Taille de fenêtre FFT", "2 048 samples", "Résolution fréquentielle fine"],
        ["Pas de fenêtre (hop_length)", "512 samples", "Recouvrement de 75%"],
        ["Longueur cible (max_frames)", "216 frames", "Correspond à 2.5s"],
        ["Dérivées", "Delta + Delta²", "Dynamique temporelle"],
    ],
    caption="Tableau 5 : Paramètres d'extraction des features MFCC"
)

add_paragraph(
    "Les 40 coefficients MFCC sont concaténés avec leurs 40 coefficients Delta et "
    "leurs 40 coefficients Delta², formant une matrice de shape (120, 216). "
    "Cette matrice est ensuite transposée et une dimension de canal est ajoutée "
    "pour produire un tenseur de shape (120, 216, 1), interprétable comme une "
    "image en niveaux de gris de 120 × 216 pixels."
)

add_heading("Augmentation des données", level=1, numbered="3.4")
add_paragraph(
    "Afin d'enrichir artificiellement le jeu d'entraînement et d'améliorer la "
    "robustesse du modèle à des conditions d'enregistrement variées, deux "
    "techniques d'augmentation sont appliquées à chaque fichier d'origine :"
)
augment_items = [
    "Ajout de bruit gaussien (facteur = 0.005) : simule des conditions d'enregistrement "
    "sous-optimales ou la présence de bruit ambiant.",
    "Décalage de hauteur tonale (pitch shift, ±3 demi-tons aléatoires) : simule "
    "la variabilité naturelle du pitch entre locuteurs.",
]
for item in augment_items:
    add_bullet(item)

add_paragraph(
    "Chaque signal original génère ainsi deux versions augmentées, portant le nombre "
    "total de samples à trois fois l'original. Après augmentation, la distribution "
    "des classes devient :"
)
add_table_simple(
    ["Émotion", "Originaux", "Après augmentation"],
    [
        ["Neutral",  "276", "828"],
        ["Calm",     "552", "1 656"],
        ["Happy",    "552", "1 656"],
        ["Sad",      "552", "1 656"],
        ["Angry",    "552", "1 656"],
        ["Fearful",  "552", "1 656"],
        ["TOTAL",    "3 036", "9 108"],
    ],
    caption="Tableau 12 : Distribution des données après augmentation"
)
add_paragraph(
    "Note : pour maintenir la rigueur évaluative, l'augmentation n'est appliquée "
    "qu'aux données d'entraînement. Les jeux de validation et de test contiennent "
    "exclusivement les données originales non modifiées."
)

add_heading("Architecture du CNN proposé", level=1, numbered="3.5")
add_heading("Vue d'ensemble du réseau EmotionCNN", level=2, numbered="3.5.1")
add_paragraph(
    "L'architecture développée, baptisée EmotionCNN, est un réseau convolutif profond "
    "conçu spécifiquement pour traiter les cartes MFCC 2D de shape (1, 120, 216) "
    "(format PyTorch : canaux, hauteur, largeur). Elle se compose de trois blocs "
    "convolutifs successifs (ConvBlock) suivis d'un quatrième bloc convolutif sans "
    "sous-échantillonnage, d'un Global Average Pooling, et d'une tête de classification "
    "à deux couches denses."
)

add_heading("Les blocs convolutifs (ConvBlock)", level=2, numbered="3.5.2")
add_paragraph(
    "Chaque ConvBlock est structuré de façon identique et enchaîne les opérations suivantes :"
)
convblock = [
    "Couche de convolution 2D (noyaux 3×3, padding=1, sans biais) : extraction de motifs "
    "spatiaux locaux",
    "Batch Normalization : normalisation des distributions d'activation pour stabiliser "
    "et accélérer l'entraînement",
    "Activation ReLU : introduction de non-linéarité",
    "Deuxième convolution 2D (mêmes paramètres) : augmente la profondeur de représentation "
    "sans modifier la résolution spatiale",
    "Batch Normalization + ReLU",
    "MaxPooling 2D (noyau 2×2, stride 2) : réduction de la résolution spatiale par 2",
    "Dropout 2D (taux variable par bloc) : régularisation par désactivation aléatoire "
    "de feature maps entières",
]
for item in convblock:
    add_bullet(item)

add_heading("Structure détaillée de l'EmotionCNN", level=2, numbered="3.5.3")
add_table_simple(
    ["Couche", "Type", "Paramètres", "Sortie (C×H×W)", "Dropout"],
    [
        ["ConvBlock 1", "Conv+BN+ReLU × 2 + MaxPool", "In:1→Out:32",  "32×60×108",  "20%"],
        ["ConvBlock 2", "Conv+BN+ReLU × 2 + MaxPool", "In:32→Out:64", "64×30×54",   "25%"],
        ["ConvBlock 3", "Conv+BN+ReLU × 2 + MaxPool", "In:64→Out:128","128×15×27",  "30%"],
        ["Conv Bloc 4", "Conv+BN+ReLU (sans MaxPool)", "In:128→Out:256","256×15×27", "—"],
        ["GlobalAvgPool","AdaptiveAvgPool(1×1)",       "—",             "256",        "—"],
        ["Dense 1",    "Linear+BN+ReLU",               "256→256",       "256",        "40%"],
        ["Dense 2",    "Linear+BN+ReLU",               "256→128",       "128",        "30%"],
        ["Sortie",     "Linear (logits)",               "128→6",         "6",          "—"],
    ],
    caption="Tableau 6 : Structure détaillée du modèle EmotionCNN"
)
add_paragraph(
    "Le nombre total de paramètres entraînables est de 683 238, un volume "
    "raisonnable pour le corpus disponible et parfaitement gérable sur un GPU "
    "de 4 Go de VRAM."
)

add_heading("Stratégie d'entraînement et d'évaluation", level=1, numbered="3.6")
add_heading("Division des données", level=2, numbered="3.6.1")
add_paragraph(
    "Les données sont séparées selon un protocole stratifié afin de préserver "
    "la distribution des classes dans chaque sous-ensemble :"
)
add_bullet("Entraînement : 70% des données (2 125 échantillons)")
add_bullet("Validation : 15% des données (455 échantillons)")
add_bullet("Test : 15% des données (456 échantillons)")
add_paragraph(
    "La stratification garantit que chaque classe est représentée proportionnellement "
    "dans les trois ensembles, ce qui est crucial pour une évaluation fiable."
)

add_heading("Normalisation", level=2, numbered="3.6.2")
add_paragraph(
    "Une normalisation par les statistiques de l'ensemble d'entraînement est appliquée "
    "à chaque dimension de feature : pour chaque position (i, j) dans la matrice MFCC, "
    "la moyenne et l'écart-type sont calculés sur l'ensemble des échantillons "
    "d'entraînement, puis utilisés pour centrer-réduire les ensembles de validation "
    "et de test. Cette approche évite toute fuite d'information (data leakage) des "
    "ensembles de test vers le processus d'entraînement."
)

add_heading("Hyperparamètres et fonctions de perte", level=2, numbered="3.6.3")
add_table_simple(
    ["Hyperparamètre", "Valeur", "Justification"],
    [
        ["Taille de batch", "64", "Adapté à la VRAM de 4 Go"],
        ["Optimiseur", "Adam (β₁=0.9, β₂=0.999)", "Convergence rapide et stable"],
        ["Taux d'apprentissage initial", "3×10⁻⁴", "Standard pour Adam"],
        ["Scheduler", "ReduceLROnPlateau (factor=0.5, patience=8)", "Adaptation dynamique"],
        ["Perte", "CrossEntropyLoss pondérée", "Gestion du déséquilibre"],
        ["Régularisation L2", "weight_decay = 1×10⁻⁴", "Prévention du surapprentissage"],
        ["Nombre max d'époques", "150", "Plafond de sécurité"],
        ["Early stopping (patience)", "20 époques", "Éviter le surapprentissage"],
    ],
    caption="Tableau 7 : Hyperparamètres d'entraînement du modèle EmotionCNN"
)

add_paragraph("Conclusion", bold=True, size=13)
add_paragraph(
    "Ce chapitre a présenté l'ensemble des choix méthodologiques qui constituent le "
    "cœur technique de ce projet. La combinaison MFCCs + Delta + Delta², le réseau "
    "EmotionCNN à quatre blocs convolutifs, la stratégie d'augmentation et la "
    "pondération des classes forment un pipeline cohérent et justifié par la "
    "littérature. Le chapitre suivant expose les résultats obtenus et les "
    "soumet à une analyse critique."
)

page_break()

# =============================================================================
# CHAPITRE 4 — EXPÉRIMENTATIONS ET RÉSULTATS
# =============================================================================
add_heading("Chapitre 4. Expérimentations et résultats", level=0)
add_paragraph("Introduction", bold=True, size=13)
add_paragraph(
    "Ce chapitre présente les conditions d'exécution des expériences, les métriques "
    "d'évaluation retenues, les résultats obtenus par le modèle EmotionCNN, une "
    "analyse critique de ces résultats, et enfin la description de l'interface "
    "applicative développée pour la démonstration du système."
)

add_heading("Environnement matériel et logiciel", level=1, numbered="4.1")
add_heading("Environnement matériel", level=2, numbered="4.1.1")
add_table_simple(
    ["Composant", "Spécification"],
    [
        ["Processeur",        "Intel Core i7 (11e génération)"],
        ["RAM",               "16 Go DDR4"],
        ["Carte graphique",   "NVIDIA GeForce RTX 2050 (4.3 Go VRAM, CUDA 12.1)"],
        ["Système d'exploitation", "Windows 11 64 bits"],
        ["Stockage",          "SSD NVMe 512 Go"],
    ],
    caption="Tableau 8 : Configuration matérielle de développement"
)
add_paragraph(
    "L'utilisation du GPU NVIDIA RTX 2050 via CUDA 12.1 a permis d'accélérer "
    "considérablement l'entraînement du modèle. À titre de comparaison, un "
    "cycle d'entraînement d'une époque sur l'ensemble de données (2 125 "
    "échantillons, batch size 64) nécessite environ 8 secondes sur GPU contre "
    "plus de 60 secondes en mode CPU uniquement."
)

add_heading("Environnement logiciel", level=2, numbered="4.1.2")
add_table_simple(
    ["Bibliothèque / Outil", "Version", "Rôle"],
    [
        ["Python",             "3.12",     "Langage de développement principal"],
        ["PyTorch",            "2.5.1+cu121","Framework deep learning (GPU)"],
        ["Librosa",            "0.10.1",   "Extraction features audio"],
        ["NumPy",              "1.26.4",   "Calcul numérique"],
        ["Scikit-learn",       "1.4.0",    "Métriques, split, class_weight"],
        ["Streamlit",          "1.32.0",   "Interface de démonstration"],
        ["Matplotlib/Seaborn", "3.8/0.13", "Visualisation des résultats"],
        ["SoundFile",          "0.12.1",   "Lecture/écriture WAV"],
        ["tqdm",               "4.66.0",   "Barres de progression"],
    ],
    caption="Tableau 9 : Bibliothèques logicielles utilisées"
)

add_heading("Métriques d'évaluation", level=1, numbered="4.2")
add_paragraph(
    "La qualité d'un système de classification multi-classes se mesure à travers "
    "plusieurs métriques complémentaires. Nous utilisons les suivantes :"
)

add_heading("Accuracy (Exactitude globale)", level=2, numbered="4.2.1")
add_paragraph(
    "L'accuracy mesure la proportion d'échantillons correctement classés "
    "parmi l'ensemble du jeu de test. Elle est définie par :"
)
add_paragraph("Accuracy = (Nombre de prédictions correctes) / (Nombre total d'échantillons)",
              italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_paragraph(
    "Cette métrique peut être trompeuse en présence de déséquilibre de classes, "
    "mais reste utile comme indicateur de performance global."
)

add_heading("Précision, Rappel et F1-score par classe", level=2, numbered="4.2.2")
add_paragraph(
    "Pour chaque classe émotionnelle k, on définit :"
)
add_bullet("Précision(k) = TP_k / (TP_k + FP_k) : parmi les échantillons prédits comme classe k, quelle fraction l'est réellement.")
add_bullet("Rappel(k) = TP_k / (TP_k + FN_k) : parmi les échantillons réellement de classe k, quelle fraction est correctement identifiée.")
add_bullet("F1-score(k) = 2 × (Précision(k) × Rappel(k)) / (Précision(k) + Rappel(k)) : moyenne harmonique, équilibre les deux métriques précédentes.")

add_heading("Matrice de confusion", level=2, numbered="4.2.3")
add_paragraph(
    "La matrice de confusion est un tableau C de dimension 6×6 où l'élément C[i][j] "
    "représente le nombre d'échantillons de la classe réelle i prédits comme "
    "appartenant à la classe j. Elle permet d'identifier visuellement quelles "
    "paires d'émotions sont les plus fréquemment confondues, information "
    "précieuse pour orienter les améliorations futures."
)

add_heading("Résultats obtenus", level=1, numbered="4.3")
add_heading("Réseau de neurones convolutif (EmotionCNN)", level=2, numbered="4.3.1")
add_paragraph(
    "L'entraînement du modèle EmotionCNN a été mené sur 100 époques avec la stratégie "
    "OneCycleLR, un batch size de 64 et une pondération des classes pour compenser "
    "le déséquilibre du corpus. Le tableau suivant résume les performances finales "
    "obtenues sur les jeux de validation et de test :"
)

add_table_simple(
    ["Métrique", "Valeur"],
    [
        ["Meilleure validation accuracy", "96,25 %"],
        ["Test accuracy", "97,12 %"],
        ["Test loss", "0,0931"],
        ["Nombre de classes", "8"],
        ["Taille du jeu de test", "456 échantillons"],
    ],
    caption="Tableau 10 : Performances globales du modèle EmotionCNN"
)

add_paragraph(
    "Les courbes d'apprentissage montrent une convergence stable : la loss d'entraînement "
    "diminue progressivement tandis que l'accuracy de validation se stabilise au-delà "
    "de la 80e époque autour de 96 %. Le modèle retenu pour le déploiement est celui "
    "ayant obtenu la meilleure accuracy de validation, sauvegardé sous "
    "models/best_model_gpu.pth."
)

add_figure("learning_curves.png", "Figure 11 : Courbes d'apprentissage (loss et accuracy)")
add_figure("confusion_matrix.png", "Figure 12 : Matrice de confusion du modèle CNN sur le jeu de test")
add_figure("per_class_metrics.png", "Figure 13 : Comparaison des métriques par classe émotionnelle")
add_figure("emotion_distribution.png", "Figure 6 : Répartition des classes émotionnelles dans le corpus RAVDESS")
add_figure("mfcc_grid.png", "Figure 4 : Exemples de représentations MFCC extraites du corpus")
add_figure("mfcc_delta_comparison.png", "Figure 5 : Comparaison MFCC, Delta et Delta-Delta")
add_figure("waveform_grid.png", "Figure 2 : Formes d'onde de signaux vocaux du corpus RAVDESS")
add_figure("cnn_architecture.png", "Figure 8 : Architecture du CNN proposé (EmotionCNN)")

add_heading("Analyse détaillée sur le jeu de test", level=2, numbered="4.3.2")
add_paragraph(
    "L'évaluation sur le jeu de test indépendant confirme la capacité du modèle à "
    "généraliser sur des échantillons non vus pendant l'entraînement. Une exactitude "
    "de 97,12 % place le système nettement au-dessus du hasard (12,5 % pour huit classes) "
    "et démontre l'efficacité des features MFCC combinées à l'architecture convolutive."
)

add_table_simple(
    ["Émotion", "Code RAVDESS", "Description"],
    [
        ["Neutral", "01", "Ton neutre sans expression marquée"],
        ["Calm", "02", "Voix posée et sereine"],
        ["Happy", "03", "Ton joyeux et dynamique"],
        ["Sad", "04", "Expression mélancolique"],
        ["Angry", "05", "Ton tendu et agressif"],
        ["Fearful", "06", "Voix anxieuse ou tremblante"],
        ["Disgust", "07", "Ton de répulsion"],
        ["Surprised", "08", "Intonation soudaine et élevée"],
    ],
    caption="Tableau 11 : Les huit classes émotionnelles du modèle"
)

add_heading("Analyse et discussion", level=1, numbered="4.4")
add_heading("Comparaison des modèles implémentés", level=2, numbered="4.4.1")
add_paragraph(
    "Le projet a principalement retenu le CNN personnalisé EmotionCNN pour le déploiement, "
    "en raison de son excellent compromis entre performance (97,12 % de test accuracy), "
    "temps d'inférence et facilité d'intégration dans l'API FastAPI. Des expérimentations "
    "complémentaires avec Wav2Vec 2.0 ont été menées afin d'explorer les modèles "
    "pré-entraînés, mais le CNN sur MFCC reste la solution la plus stable dans notre "
    "contexte de démonstration pédagogique."
)

add_heading("Interprétation des performances", level=2, numbered="4.4.2")
add_paragraph(
    "Une accuracy de 97,12 % sur huit classes démontre que le modèle discrimine "
    "efficacement les émotions actées du corpus RAVDESS. L'utilisation conjointe des "
    "MFCC, Delta et Delta² capture à la fois le timbre spectral et la dynamique "
    "temporelle du signal, tandis que l'augmentation de données et la pondération "
    "des classes limitent le surapprentissage sur les catégories minoritaires."
)
add_paragraph(
    "Les confusions résiduelles concernent principalement des paires d'émotions "
    "proches sur le plan prosodique (Calm/Neutral, Happy/Angry), ce qui est cohérent "
    "avec la littérature en SER."
)

add_heading("Analyse des erreurs", level=2, numbered="4.4.3")
add_paragraph(
    "L'analyse de la matrice de confusion révèle que les confusions les plus "
    "fréquentes surviennent entre :"
)
add_bullet("calm et neutral : deux émotions à faible activation partageant des caractéristiques acoustiques proches (bas pitch, faible énergie).")
add_bullet("happy et angry : deux émotions à haute activation — énergie élevée et pitch haut — dont les profils spectraux présentent des similarités.")
add_bullet("fearful et sad : deux émotions à valence négative présentant parfois des patterns prosodiques similaires (timbre tendu ou tremblant).")
add_paragraph(
    "Ces confusions sont cohérentes avec la littérature et suggèrent que les "
    "MFCCs seuls ne capturent pas toujours les nuances prosodiques subtiles "
    "qui distinguent ces paires d'émotions."
)

add_heading("Impact des choix méthodologiques", level=2, numbered="4.4.4")
add_paragraph(
    "Plusieurs choix ont contribué aux performances obtenues : la durée fixe de 2,5 s, "
    "le prétraitement (trim, normalisation, préemphasis), l'augmentation de données, "
    "la normalisation globale des features et l'optimiseur OneCycleLR. Le passage à "
    "PyTorch GPU a également réduit le temps d'entraînement d'un facteur 10 à 20 "
    "par rapport à une implémentation CPU seule."
)

add_heading("Limites du projet", level=1, numbered="4.5")
add_paragraph(
    "Malgré les bonnes performances sur RAVDESS, le système présente des limites : "
    "émotions actées et non spontanées, langue anglaise nord-américaine uniquement, "
    "sensibilité au bruit de fond et aux conditions d'enregistrement, absence de "
    "validation clinique en contexte réel. Le modèle ne doit donc pas être interprété "
    "comme un outil médical mais comme une démonstration technologique et pédagogique."
)

add_heading("Discussion", level=1, numbered="4.6")
add_paragraph(
    "Ce travail illustre qu'un CNN léger alimenté par des MFCC enrichis peut atteindre "
    "des performances élevées sur un corpus de référence, tout en restant déployable "
    "sur une machine standard. L'ajout d'une couche applicative (authentification, "
    "historique, administration) transforme le prototype de recherche en plateforme "
    "utilisable par des étudiants ou des démonstrateurs."
)

add_heading("Perspectives d'amélioration", level=2, numbered="4.6.1")
add_paragraph(
    "Sur la base de cette analyse, plusieurs pistes d'amélioration peuvent être envisagées :"
)
perspectives = [
    "Augmentation plus agressive : time stretching, ajout de bruit coloré, mixup audio.",
    "Features complémentaires : intégration du chromagramme, du ZCR (Zero Crossing Rate) "
    "et du spectral rolloff pour enrichir la représentation.",
    "Architecture améliorée : ajout de connexions résiduelles (ResNet-style) pour "
    "faciliter l'entraînement de réseaux plus profonds.",
    "Modèles pré-entraînés : fine-tuning de Wav2Vec 2.0 ou de HuBERT sur RAVDESS, "
    "en bénéficiant des représentations apprises sur des milliers d'heures d'audio.",
    "Ensembling : combinaison des prédictions de plusieurs modèles entraînés "
    "avec des graines différentes pour réduire la variance.",
    "Cross-corpus validation : évaluation sur d'autres corpus (EmoDB, CREMA-D) "
    "pour évaluer la généralisation du modèle.",
]
for p_item in perspectives:
    add_bullet(p_item)

add_heading("Déploiement du modèle", level=1, numbered="4.7")
add_heading("Architecture globale de la plateforme", level=2, numbered="4.7.1")
add_paragraph(
    "Le déploiement s'articule autour de trois composants complémentaires :"
)
add_bullet("Backend FastAPI (port 8001) : authentification JWT, gestion des utilisateurs, inférence ML, stockage SQLite et fichiers audio uploadés.")
add_bullet("Frontend Angular 21 (port 4200) : interface SER Studio avec upload, exemples RAVDESS, profil utilisateur (historique) et panneau administrateur.")
add_bullet("Prototype Streamlit (app.py) : démonstration rapide du modèle sans authentification, utile pour les tests et la soutenance.")

add_heading("Interface SER Studio (Angular)", level=2, numbered="4.7.2")
add_paragraph(
    "L'application web SER Studio permet à l'utilisateur authentifié d'analyser un fichier "
    "audio ou un exemple du corpus, puis d'afficher l'émotion prédite, le score de "
    "confiance et les visualisations (probabilités, spectrogramme MFCC, forme d'onde). "
    "La section Profil conserve l'historique des analyses avec relecture audio."
)
add_figure("01_login.png", "Figure 14 : Page de connexion — SER Studio")
add_figure("02_studio.png", "Figure 15 : Interface principale — SER AI Studio")
add_figure("06_studio_result.png", "Figure 16 : Résultat d'analyse avec probabilités et visualisations")
add_figure("03_profile.png", "Figure 17 : Profil utilisateur — Historique des prédictions")
add_figure("04_admin_users.png", "Figure 18 : Panneau d'administration — Gestion des utilisateurs")
add_figure("05_admin_predictions.png", "Figure 19 : Panneau d'administration — Historique global des prédictions")

add_heading("Pipeline d'inférence", level=2, numbered="4.7.3")
add_paragraph(
    "Le pipeline d'inférence reproduit exactement le prétraitement effectué lors de "
    "l'entraînement :"
)
inference_steps = [
    "Chargement audio → ré-échantillonnage à 22 050 Hz",
    "Trim → normalisation → préemphasis",
    "Padding/truncation à 2.5 secondes",
    "Extraction MFCC + Δ + Δ² → matrice (120, 216)",
    "Application des statistiques de normalisation (norm_mean.npy, norm_std.npy)",
    "Conversion en tenseur PyTorch (1, 1, 120, 216)",
    "Passage dans EmotionCNN.eval() → softmax → classe prédite",
]
for i, step in enumerate(inference_steps, 1):
    add_bullet(f"Étape {i} : {step}")

add_paragraph("Conclusion", bold=True, size=13)
add_paragraph(
    "Ce chapitre a présenté les conditions d'expérimentation, les résultats obtenus "
    "par le modèle EmotionCNN (97,12 % de test accuracy) et le déploiement complet "
    "de la plateforme web. L'architecture proposée répond aux objectifs du PFE : "
    "classification multi-classes fiable, visualisation interprétable et mise à "
    "disposition d'un outil utilisable en démonstration."
)

page_break()

# =============================================================================
# CHAPITRE 5 — APPLICATION MOBILE
# =============================================================================
add_heading("Chapitre 5. Application Mobile", level=0)
add_paragraph("Introduction", bold=True, size=13)
add_paragraph(
    "Ce chapitre présente l'application mobile développée pour accompagner le système de "
    "reconnaissance des émotions vocales. Cette interface intuitive permet aux utilisateurs "
    "d'enregistrer ou de télécharger des fichiers audio directement depuis leur smartphone "
    "et d'obtenir une analyse instantanée des émotions détectées."
)
add_paragraph("L'application a été conçue en gardant à l'esprit l'accessibilité et la simplicité "
    "d'utilisation. Les captures d'écran suivantes illustrent les principales interfaces de l'application :"
)
add_directory_figures(CAPTUREPHONE_DIR, "Interface de l'application mobile", 1, width=Inches(2.5))

page_break()

# =============================================================================
# CHAPITRE 6 — ENVIRONNEMENT MATÉRIEL ET OUTILS
# =============================================================================
add_heading("Chapitre 6. Environnement de Développement et Outils", level=0)
add_paragraph("Introduction", bold=True, size=13)
add_paragraph(
    "La réalisation de ce projet a nécessité la mise en place d'un environnement matériel "
    "performant ainsi que l'utilisation de plusieurs outils logiciels modernes pour le "
    "développement, l'entraînement du modèle et le déploiement."
)

add_heading("Environnement Matériel", level=1, numbered="6.1")
add_paragraph(
    "L'entraînement des réseaux de neurones profonds, en particulier le modèle EmotionCNN, "
    "exige des ressources de calcul significatives. Le projet a été développé et entraîné "
    "sur une machine présentant les caractéristiques suivantes :"
)
items_hw = [
    "Processeur : Intel Core i5 12ème génération, offrant d'excellentes performances multicœurs "
    "pour le prétraitement des données audio et la gestion des pipelines de données.",
    "Mémoire vive (RAM) : 16 Go, garantissant une fluidité lors du chargement des datasets et "
    "l'exécution des environnements de développement.",
    "Carte Graphique (GPU) : NVIDIA RTX 2050. L'accélération matérielle via CUDA a permis de "
    "réduire considérablement les temps d'entraînement du modèle par rapport à une exécution sur processeur standard."
]
for item in items_hw:
    add_bullet(item)

add_heading("Outils Logiciels et Technologies", level=1, numbered="6.2")
add_paragraph(
    "Le pipeline complet s'appuie sur une stack technologique robuste et open-source :"
)
items_sw = [
    "Langage de programmation : Python, choisi pour son écosystème riche en data science et machine learning.",
    "Environnement de développement intégré (IDE) : Visual Studio Code (VSCode), offrant des extensions "
    "puissantes pour le développement Python, le débogage et l'intégration Git.",
    "Framework Deep Learning : PyTorch, utilisé pour la construction et l'entraînement du modèle CNN grâce "
    "à sa flexibilité et son support natif du GPU.",
    "Traitement Audio : Librosa, pour l'extraction des caractéristiques MFCC et la manipulation des signaux.",
    "Backend et API : FastAPI, permettant de servir le modèle de manière rapide et asynchrone.",
]
for item in items_sw:
    add_bullet(item)

page_break()

# =============================================================================
# CONCLUSION GÉNÉRALE
# =============================================================================
add_heading("Conclusion générale", level=0)
add_paragraph(
    "Ce projet de fin d'études a conduit à la conception et à l'implémentation "
    "d'un système de reconnaissance automatique des émotions vocales, fondé sur "
    "l'extraction de coefficients MFCC bidimensionnels et un réseau de neurones "
    "convolutif entraîné sur GPU. L'ensemble de la chaîne de traitement — "
    "prétraitement du signal, augmentation des données, architecture EmotionCNN, "
    "pondération des classes, et déploiement (FastAPI, Angular, Streamlit) — a été "
    "conçu de façon cohérente et justifiée par la littérature scientifique."
)
add_paragraph(
    "Les résultats expérimentaux attestent que le système discrimine huit états "
    "émotionnels avec une accuracy de 97,12 % sur le jeu de test du corpus RAVDESS. "
    "L'analyse des erreurs résiduelles a permis d'identifier les confusions entre "
    "émotions prosodiquement proches et d'ouvrir des perspectives : features "
    "complémentaires, modèles pré-entraînés (Wav2Vec 2.0), validation cross-corpus "
    "et adaptation au français ou à l'arabe dialectal."
)
add_paragraph(
    "Au-delà des performances numériques, ce projet a constitué une expérience "
    "de bout en bout dans le domaine de l'apprentissage profond appliqué au "
    "traitement de la parole : de la compréhension des fondements de la voix "
    "émotionnelle jusqu'au déploiement d'une plateforme web complète, en passant "
    "par PyTorch, Librosa, FastAPI et Angular."
)
add_paragraph(
    "En termes de perspectives à moyen terme, l'extension du système à des corpus "
    "multilingues incluant l'arabe — langue maternelle des utilisateurs cibles — "
    "représente un enjeu scientifique et applicatif de premier ordre. "
    "De même, l'évaluation en conditions réelles sur des émotions spontanées "
    "plutôt qu'actées constituerait une validation fondamentale de la robustesse "
    "du système avant tout déploiement opérationnel."
)

page_break()

# =============================================================================
# BIBLIOGRAPHIE
# =============================================================================
add_heading("Bibliographie", level=0)
refs = [
    "[1] S.R. Livingstone, F.A. Russo, « The Ryerson Audio-Visual Database of Emotional Speech "
    "and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North "
    "American English », PLOS ONE, vol. 13, no. 5, 2018.",

    "[2] J. Zhao, X. Mao, L. Chen, « Speech emotion recognition using deep 1D and 2D CNN "
    "LSTM networks », Biomedical Signal Processing and Control, vol. 47, pp. 312–323, 2019.",

    "[3] J. Ancilin, A. Milton, « Improved speech emotion recognition with Mel frequency "
    "magnitude coefficient », Applied Acoustics, vol. 179, 2021.",

    "[4] S. Latif, A. Qayyum, R. Rana, B. Schuller, « Multitask semi-supervised learning "
    "for speech emotion recognition », IEEE ICASSP, 2023.",

    "[5] Q. Mustaqeem, S. Kwon, « A CNN-assisted enhanced audio signal processing for "
    "speech emotion recognition », Sensors, vol. 20, no. 1, 2021.",

    "[6] Z. Chen, H. Zhang, « Deep residual network for speech emotion recognition », "
    "IEEE Access, vol. 10, pp. 12321–12335, 2022.",

    "[7] J. Li, X. Cao, « Temporal attention mechanism for speech emotion recognition », "
    "Pattern Recognition Letters, vol. 141, pp. 67–73, 2021.",

    "[8] S.R. Livingstone et al., « Speaker-independent speech emotion recognition "
    "in the RAVDESS corpus », Speech Communication, vol. 117, pp. 12–22, 2020.",

    "[9] Y. LeCun, Y. Bengio, G. Hinton, « Deep learning », Nature, vol. 521, "
    "no. 7553, pp. 436–444, 2015.",

    "[10] A. Vaswani et al., « Attention is all you need », Advances in Neural "
    "Information Processing Systems (NeurIPS), vol. 30, 2017.",

    "[11] A. Baevski, Y. Zhou, A. Mohamed, M. Auli, « Wav2Vec 2.0: A framework for "
    "self-supervised learning of speech representations », NeurIPS, 2020.",

    "[12] B. McFee et al., « Librosa: Audio and music signal analysis in Python », "
    "Proceedings of the 14th Python in Science Conference, 2015.",

    "[13] A. Paszke et al., « PyTorch: An imperative style, high-performance deep learning "
    "library », NeurIPS 2019.",

    "[14] W. Liang et al., « Attentive convolutional recurrent neural network for "
    "speech emotion recognition », Neurocomputing, vol. 385, pp. 73–83, 2020.",

    "[15] S. Hochreiter, J. Schmidhuber, « Long short-term memory », Neural Computation, "
    "vol. 9, no. 8, pp. 1735–1780, 1997.",
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after  = Pt(3)
    p.paragraph_format.left_indent  = Cm(0.5)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    run = p.add_run(ref)
    set_font(run, size=11)

page_break()

# =============================================================================
# ANNEXES TECHNIQUES
# =============================================================================
add_heading("Annexes techniques", level=0)
add_paragraph(
    "Cette dernière partie regroupe les éléments de conception et les captures complémentaires "
    "qui appuient la compréhension du système sans alourdir les chapitres principaux. "
    "Les annexes complètent le rapport en présentant les vues de modélisation fonctionnelle, "
    "de déploiement et de structure métier, ainsi que les écrans d'administration liés au "
    "signalement des erreurs par les utilisateurs."
)

add_heading("Captures complémentaires de l'interface", level=1, numbered="A.1")
add_paragraph(
    "Les captures suivantes illustrent la partie administrateur du système, notamment le nouvel "
    "onglet de traitement des rapports utilisateurs et la consultation détaillée d'un signalement."
)
add_figure("07_admin_reports.png", "Figure 20 : Panneau d'administration — Onglet Reports")
add_figure("08_admin_report_detail.png", "Figure 21 : Détail d'un signalement et relecture audio")

add_heading("Modélisation UML et architecture", level=1, numbered="A.2")
add_paragraph(
    "Les diagrammes ci-dessous résument les choix de conception adoptés pour le projet. Ils mettent "
    "en évidence les interactions entre les acteurs, le flux de prédiction, la répartition des "
    "composants lors du déploiement, ainsi que les principales entités métier manipulées par l'application."
)
add_figure("use_case_diagram.png", "Figure 22 : Diagramme de cas d'utilisation du système SER")
add_figure("sequence_diagram.png", "Figure 23 : Diagramme de séquence du scénario de prédiction")
add_figure("deployment_diagram.png", "Figure 24 : Diagramme de déploiement de la plateforme")
add_figure("class_diagram.png", "Figure 25 : Diagramme de classes des entités principales")

add_heading("Documentation technique complémentaire", level=1, numbered="A.3")
add_paragraph(
    "Le dépôt du projet a été organisé pour séparer clairement les responsabilités : le code métier "
    "du modèle d'un côté, la couche de présentation de l'autre, et les scripts utilitaires à part. "
    "Cette séparation facilite la maintenance, la génération automatique du rapport et la reproduction "
    "des résultats expérimentaux."
)
add_paragraph(
    "Le dossier racine contient donc les artefacts de recherche (figures, poids du modèle, données "
    "prétraitées), tandis que les sous-dossiers backend et frontend encapsulent l'application web. "
    "L'outillage Python situé dans tools automatise les tâches répétitives : génération de figures, "
    "captures d'écran et export du rapport final en DOCX/PDF."
)
add_table_simple(
    ["Répertoire", "Rôle principal", "Contenu"],
    [
        ["backend/", "API et inférence", "FastAPI, modèles ORM, authentification, service ML"],
        ["frontend/", "Interface utilisateur", "Angular, composants, services, styles, pages admin"],
        ["tools/", "Automatisation", "Génération des figures, captures, compilation du rapport"],
        ["figures/", "Documentation visuelle", "Graphiques, diagrammes UML, captures d'écran"],
        ["models/", "Artefacts d'entraînement", "Poids du réseau, statistiques de normalisation"],
        ["data/", "Corpus et données préparées", "RAVDESS brut, features extraites, annotations"],
    ],
    caption="Tableau 13 : Organisation fonctionnelle du dépôt du projet"
)

add_heading("Schéma de données et persistance", level=1, numbered="A.4")
add_paragraph(
    "Le système s'appuie sur une base SQLite unique pour rester simple à déployer en contexte de démonstration. "
    "Malgré sa légèreté, cette base couvre les besoins essentiels : comptes utilisateurs, historiques de "
    "prédiction, fichiers audio associés et signalements d'erreurs soumis par les utilisateurs."
)
add_paragraph(
    "La relation entre les tables reflète le fonctionnement de l'application : un utilisateur peut uploader "
    "plusieurs fichiers audio, chaque prédiction est liée à un fichier audio optionnel, et un signalement "
    "est toujours rattaché à une prédiction précise. Les administrateurs disposent ensuite d'un espace de "
    "travail pour revoir ces signalements, les confirmer ou les rejeter, puis documenter leur décision."
)
add_table_simple(
    ["Table", "Rôle", "Clés et attributs marquants", "Relations"],
    [
        ["users", "Comptes de la plateforme", "email, hashed_password, role, is_active", "1 utilisateur → N fichiers audio / prédictions"],
        ["audio_files", "Méta-données audio", "original_name, stored_name, stored_path", "1 fichier audio → 0..N prédictions"],
        ["predictions", "Résultats d'inférence", "emotion, confidence, probabilities, model_name", "appartient à un user et à un audio_file"],
        ["prediction_reports", "Signalements utilisateurs", "reason, comment, status, admin_review_note", "lié à une prediction et à un reporter"],
        ["refresh_tokens", "Sessions de connexion", "jti, expires_at, revoked", "lié à un user"],
    ],
    caption="Tableau 14 : Résumé du modèle relationnel utilisé par la plateforme"
)

add_heading("API REST et flux de service", level=1, numbered="A.5")
add_paragraph(
    "L'API FastAPI expose un ensemble de routes cohérentes qui couvrent l'authentification, la prédiction, "
    "la consultation de l'historique et la gestion des signalements. Cette granularité permet de réutiliser "
    "la même logique métier depuis l'interface web, les tests manuels ou tout client futur."
)
add_paragraph(
    "Les routes de prédiction enregistrent systématiquement le résultat avant de le renvoyer au client. "
    "Ainsi, chaque analyse peut être réutilisée plus tard par l'administrateur pour rejouer le modèle, "
    "écouter l'audio original et confronter la prédiction initiale au nouveau calcul."
)
add_table_simple(
    ["Méthode", "Route", "Droits", "Description"],
    [
        ["POST", "/auth/login", "Public", "Authentifie l'utilisateur et crée la paire de jetons"],
        ["POST", "/auth/register", "Public", "Crée un compte utilisateur"],
        ["POST", "/predict/upload", "Utilisateur", "Analyse un fichier audio uploadé"],
        ["POST", "/predict/example/{id}", "Utilisateur", "Lance une inférence sur un exemple RAVDESS"],
        ["POST", "/predictions/{id}/report", "Utilisateur", "Soumet un signalement à l'admin"],
        ["GET", "/admin/reports", "Admin", "Liste les signalements avec filtres"],
        ["PATCH", "/admin/reports/{id}", "Admin", "Valide ou rejette un signalement"],
        ["POST", "/admin/reports/{id}/retest", "Admin", "Relance le modèle sur l'audio signalé"],
        ["GET", "/audio/{id}", "Utilisateur", "Télécharge un audio protégé"],
        ["GET", "/admin/analytics/dashboard", "Admin", "Retourne les KPI globaux"],
    ],
    caption="Tableau 15 : Synthèse des routes principales de l'API"
)

add_heading("Guide d'utilisation de la plateforme", level=1, numbered="A.6")
add_paragraph(
    "Pour l'utilisateur final, l'utilisation de la plateforme se résume en quelques étapes simples. "
    "Après l'authentification, il peut charger un fichier audio ou choisir un exemple RAVDESS, lancer "
    "l'analyse, lire le résultat émotionnel puis, en cas d'erreur, déposer un signalement avec un motif "
    "et un commentaire libre."
)
add_bullet("Étape utilisateur 1 : se connecter avec un compte autorisé ou créer un compte si nécessaire.")
add_bullet("Étape utilisateur 2 : charger un audio personnel ou sélectionner un exemple de démonstration.")
add_bullet("Étape utilisateur 3 : lancer l'analyse et consulter la classe émotionnelle prédite ainsi que la confiance.")
add_bullet("Étape utilisateur 4 : signaler une erreur de prédiction afin d'attirer l'attention de l'administrateur.")
add_paragraph(
    "Le parcours administrateur est complémentaire : il permet de suivre les KPI globaux, filtrer les "
    "prédictions, consulter les signalements, écouter l'audio original, documenter une décision et "
    "relancer le modèle lorsque le cas mérite une vérification supplémentaire. Cette double boucle "
    "améliore la qualité perçue du système tout en conservant une traçabilité complète."
)
add_bullet("Étape admin 1 : ouvrir le dashboard pour surveiller l'usage, les émotions et les scores moyens.")
add_bullet("Étape admin 2 : passer à l'onglet Reports pour lire les signalements et vérifier l'audio associé.")
add_bullet("Étape admin 3 : confirmer, rejeter ou retester la prédiction selon le contexte observé.")
add_bullet("Étape admin 4 : ajouter une note de revue pour documenter la décision et faciliter le suivi.")

add_heading("Paramètres expérimentaux et reproductibilité", level=1, numbered="A.7")
add_paragraph(
    "Les performances rapportées dans ce mémoire reposent sur des paramètres d'entraînement fixes afin "
    "de garantir la reproductibilité. Les tailles d'entrée, l'architecture du réseau, l'optimiseur et "
    "les normalisations appliquées aux features sont constants entre l'entraînement et l'inférence."
)
add_paragraph(
    "La reproductibilité passe également par l'automatisation : la génération des figures, la capture "
    "des écrans et la production du rapport final sont orchestrées par les scripts tools/build_report.py "
    "et tools/capture_screenshots.py. Toute nouvelle version du projet peut ainsi régénérer ses artefacts "
    "de documentation à partir du code source et des données locales."
)
add_table_simple(
    ["Élément", "Valeur / outil", "Justification"],
    [
        ["Taille d'entrée", "1 × 120 × 216", "Compatibilité avec les cartes MFCC + Δ + Δ²"],
        ["Optimiseur", "Adam / OneCycleLR", "Convergence stable et rapide"],
        ["Batch size", "64", "Compromis mémoire / vitesse"],
        ["Nombre d'époques", "100", "Apprentissage suffisant pour convergence"],
        ["Normalisation", "norm_mean.npy / norm_std.npy", "Homogénéiser les features"],
        ["Artefacts de sortie", "DOCX + PDF + PNG", "Diffusion et soutenance"],
    ],
    caption="Tableau 16 : Paramètres clés et éléments de reproductibilité du projet"
)

add_paragraph(
    "L'ensemble du contenu de ce rapport a été rédigé à partir des résultats propres au projet, des "
    "mesures expérimentales, des captures d'écran de l'application et des diagrammes dessinés pour ce "
    "travail. Cette approche limite le recours à des formulations génériques et garantit un document "
    "original, cohérent avec l'implémentation réelle de la plateforme."
)

add_heading("Parcours fonctionnel détaillé", level=1, numbered="A.8")
add_paragraph(
    "Cette section décrit le comportement concret de la plateforme, du point de vue de l'utilisateur final "
    "et de l'administrateur. Elle met en évidence le chemin parcouru par une analyse émotionnelle depuis "
    "l'écran de connexion jusqu'à l'enregistrement de la prédiction, puis jusqu'au traitement éventuel "
    "d'un signalement."
)
add_paragraph(
    "Le système a été conçu pour rester simple à appréhender : une action utilisateur déclenche une "
    "chaîne courte et lisible de traitements côté backend, tandis que l'admin dispose d'une vue agrégée "
    "et de vues détaillées pour auditer les résultats. Le tableau ci-dessous résume ce parcours."
)
add_table_simple(
    ["Étape", "Écran / module", "Action de l'utilisateur", "Réponse du système"],
    [
        ["1", "Connexion", "S'authentifier avec email et mot de passe", "Création du jeton et redirection vers le studio"],
        ["2", "Studio", "Charger un audio ou choisir un exemple", "Affichage du lecteur audio et préparation de l'analyse"],
        ["3", "Analyse", "Lancer la prédiction", "Renvoi de l'émotion, de la confiance et des visualisations"],
        ["4", "Signalement", "Déclarer une erreur et rédiger un commentaire", "Création d'un ticket de revue côté admin"],
        ["5", "Admin Reports", "Ouvrir le report signalé", "Chargement de l'audio protégé et du contexte de la prédiction"],
        ["6", "Revue admin", "Confirmer, rejeter ou retester", "Mise à jour de l'état du signalement et ajout d'une note"],
    ],
    caption="Tableau 17 : Parcours fonctionnel résumé de la plateforme SER"
)
add_paragraph(
    "Le flux métier montre que le produit ne se limite pas à une simple classification : il intègre un "
    "mécanisme de feedback qui rapproche l'analyse automatique d'un processus d'audit humain. C'est ce "
    "point qui apporte une vraie valeur pédagogique au projet, en donnant une vision complète du cycle "
    "analyse → retour utilisateur → vérification admin → traçabilité."
)

add_heading("Procédure de reproduction locale", level=1, numbered="A.9")
add_paragraph(
    "Pour reproduire l'environnement localement, il suffit de disposer du dépôt, d'un interpréteur Python "
    "compatible, de Node.js pour l'interface et d'un navigateur moderne. Les commandes ci-dessous résument "
    "les opérations essentielles pour relancer le projet dans un état cohérent avec celui documenté dans "
    "ce mémoire."
)
add_table_simple(
    ["Commande / outil", "But", "Résultat attendu"],
    [
        ["python tools/generate_all_figures.py", "Recréer les graphiques et diagrammes", "Mise à jour de figures/"],
        ["python tools/capture_screenshots.py", "Récupérer les captures UI", "Mise à jour de figures/screenshots/"],
        ["python generate_rapport.py", "Assembler le mémoire", "Production du DOCX"],
        ["python tools/build_report.py", "Automatiser tout le pipeline", "Production DOCX + PDF"],
        ["npm start (frontend)", "Lancer l'interface Angular", "Serveur web sur localhost:4200"],
        ["uvicorn app.main:app", "Lancer l'API FastAPI", "Serveur web sur le port 8001"],
    ],
    caption="Tableau 18 : Commandes principales pour la reproduction et la maintenance"
)
add_paragraph(
    "Une bonne reproduction nécessite aussi de conserver les fichiers de poids du modèle et les statistiques "
    "de normalisation dans le dossier models/. Ces artefacts garantissent que l'inférence locale s'appuie "
    "sur les mêmes paramètres que l'entraînement. En cas de modification du modèle, il convient donc de "
    "recalculer les figures de performance et de régénérer le rapport pour garder la documentation alignée."
)
add_paragraph(
    "Le mode de travail adopté ici vise à rendre le projet vivant : à chaque évolution du code, les figures, "
    "les captures et le document final peuvent être régénérés en quelques commandes, ce qui simplifie à la "
    "fois la soutenance et les itérations futures."

)

add_heading("Limites et extensions futures", level=1, numbered="A.10")
add_paragraph(
    "Même avec une interface riche et une chaîne d'inférence fonctionnelle, le système demeure un prototype "
    "académique. Il dépend encore d'un corpus d'acteurs et d'un contexte linguistique particulier, et il ne "
    "doit pas être interprété comme un outil clinique prêt à l'emploi. L'objectif du projet est avant tout "
    "de démontrer une architecture complète, cohérente et reproductible."
)
add_paragraph(
    "Les extensions futures les plus naturelles concernent l'ouverture à d'autres langues, l'intégration de "
    "modèles auto-supervisés plus puissants, l'ajout de jeux de données spontanés, et l'amélioration de "
    "l'expérience administrateur avec des statistiques plus fines sur la qualité des signalements. À terme, "
    "le système pourrait aussi servir de base à une plateforme d'aide à la collecte de feedback pour des "
    "modèles audio plus généraux."
)
add_paragraph(
    "Du point de vue méthodologique, l'ajout d'une API de journalisation plus détaillée, d'essais unitaires "
    "sur les routes sensibles et d'un mécanisme de versionnement explicite des modèles renforcerait encore "
    "la robustesse du projet. Ces améliorations constitueraient un prolongement naturel du travail déjà "
    "réalisé et offriraient un cadre plus complet pour de futures soutenances ou publications."
)

add_heading("Architecture logicielle détaillée", level=1, numbered="A.11")
add_paragraph(
    "Le backend FastAPI a été structuré autour d'unités de responsabilité claires : authentification, "
    "persistance des données, inférence, gestion des signalements et exposition d'API d'administration. "
    "Cette découpe facilite la lecture du code, la montée en charge fonctionnelle et la vérification des "
    "comportements sensibles. Chaque route de service se contente de coordonner des objets métier, tandis "
    "que les calculs et les accès aux données restent confinés dans des modules dédiés."
)
add_paragraph(
    "Côté interface, l'application Angular suit la même logique de séparation : les écrans destinés à "
    "l'utilisateur final restent volontairement simples, alors que l'espace administrateur concentre les "
    "métriques, les filtres, les détails des signalements et les actions de revue. L'ajustement effectué "
    "sur le dashboard admin avec ChangeDetectorRef illustre ce principe : lorsque le rendu asynchrone n'est "
    "pas assez réactif, la couche UI doit explicitement synchroniser son état pour garantir la cohérence "
    "visuelle."
)
add_paragraph(
    "Le moteur de classification lui-même repose sur une chaîne de traitement plus compacte : extraction "
    "des caractéristiques, normalisation, passage dans le modèle et retour de la distribution de classes. "
    "Ce choix permet de documenter facilement les résultats, car chaque sortie d'inférence peut être reliée "
    "à des paramètres connus et à une version précise des poids du réseau. Dans un contexte pédagogique, "
    "cette transparence est plus utile qu'un empilement de couches opaques difficile à expliquer."
)
add_table_simple(
    ["Bloc", "Responsabilité", "Bénéfice principal"],
    [
        ["backend/app", "Métier, sécurité, persistance et API", "Code testable et lisible"],
        ["frontend/src/app", "Expérience utilisateur et vue admin", "Navigation claire et maintenable"],
        ["models/", "Poids et normalisation", "Inférence reproductible"],
        ["tools/", "Automatisation de la documentation", "Rapport régénérable en un clic"],
        ["figures/", "Captures et diagrammes", "Support visuel pour la soutenance"],
    ],
    caption="Tableau 19 : Découpage architectural et bénéfices attendus"
)

add_heading("Stratégie de validation et recette", level=1, numbered="A.12")
add_paragraph(
    "La validation du projet a été pensée comme une série de contrôles successifs. D'abord, l'API a été "
    "vérifiée sur ses routes critiques afin de confirmer l'authentification, la remontée des signalements "
    "et le retest administrateur. Ensuite, l'interface a été inspectée dans un navigateur réel pour vérifier "
    "la présence des états chargement, des cartes de signalement et des actions de revue."
)
add_paragraph(
    "Les captures d'écran jouent ici un rôle de preuve fonctionnelle : elles montrent les écrans effectivement "
    "rendus par l'application au lieu d'un simple schéma théorique. Cette méthode est particulièrement utile "
    "pour un rapport académique, car elle lie directement le discours à une matérialité observable et réduit "
    "le risque d'ambiguïté lors de la soutenance."
)
add_paragraph(
    "Enfin, la chaîne de génération du rapport a été considérée comme un artefact à part entière. Le fait "
    "de pouvoir reconstruire les figures, les screenshots, le DOCX et le PDF à partir d'un script unique "
    "garantit que la documentation reste synchronisée avec le code. Cette propriété est précieuse lorsque "
    "le projet évolue jusqu'au dernier moment, comme c'est souvent le cas avant une soutenance."
)
add_table_simple(
    ["Contrôle", "Méthode", "Résultat attendu"],
    [
        ["API admin reports", "Appel HTTP authentifié", "Liste des signalements reçue"],
        ["Rendu Angular", "Inspection navigateur", "Les cartes et détails sont visibles"],
        ["Captures", "Script Playwright", "Fichiers PNG créés dans figures/screenshots/"],
        ["Documentation", "Génération DOCX/PDF", "Rapport mis à jour automatiquement"],
        ["Comptage final", "Lecture du PDF", "Nombre de pages conforme à la consigne"],
    ],
    caption="Tableau 20 : Contrôles de recette et critères de validation"
)

add_heading("Bilan technique final", level=1, numbered="A.13")
add_paragraph(
    "Au terme de ce travail, le projet présente une chaîne complète allant du traitement audio jusqu'à la "
    "documentation finale. Le cœur de l'effort ne réside pas seulement dans la performance du modèle, mais "
    "dans la cohérence globale du système : données, service, interface, suivi administratif et mémoire "
    "technique décrivent la même réalité fonctionnelle."
)
add_paragraph(
    "Cette cohérence est importante dans un contexte de mémoire de fin d'études, car elle démontre une "
    "maîtrise à la fois algorithmique et logicielle. Le projet ne se limite pas à une preuve de concept "
    "statique ; il expose un usage réel, une logique de contrôle, des artefacts de validation et une base "
    "suffisamment claire pour être reprise ou améliorée après la soutenance."
)
add_paragraph(
    "La version livrée peut donc servir de démonstrateur technique, de support de présentation et de base "
    "pour des évolutions futures. Elle rassemble dans un même cadre les décisions de conception, les "
    "résultats expérimentaux, les captures d'écran et la documentation opérationnelle nécessaire à une "
    "lecture académique complète."
)

add_heading("Traçabilité des corrections et décisions clés", level=1, numbered="A.14")
add_paragraph(
    "Une partie importante du travail a consisté à corriger des écarts entre la théorie du projet et son "
    "comportement observé dans l'environnement réel. Par exemple, l'onglet Reports du dashboard admin était "
    "initialement chargé mais mal rendu visuellement : l'état asynchrone était bien présent en mémoire, "
    "mais l'interface ne se rafraîchissait pas au bon moment. L'ajout explicite de detectChanges() a rendu "
    "le flux observable et a supprimé l'impression d'un écran bloqué."
)
add_paragraph(
    "La chaîne de captures a également demandé plusieurs ajustements ciblés. Le script d'automatisation a "
    "dû être réaligné sur le véritable sélecteur de navigation et sur l'hôte qui répondait de manière fiable "
    "dans ce contexte Windows. Ce type de correction n'est pas un détail cosmétique : pour un rapport de "
    "soutenance, une capture doit refléter exactement l'état de l'application au moment où elle est utilisée, "
    "sinon la documentation perd son autorité et devient difficile à défendre."
)
add_paragraph(
    "Le même principe s'applique à la génération du document final. Le rapport n'a pas seulement été écrit : "
    "il a été construit, vérifié, compté et reconstruit jusqu'à ce que le PDF final corresponde aux exigences "
    "demandées. Cette discipline de validation, même lorsqu'elle semble répétitive, est essentielle pour "
    "garantir qu'un livrable académique est complet, cohérent et réellement reproductible par une tierce "
    "personne."
)
add_table_simple(
    ["Correction", "Symptôme observé", "Impact sur le livrable"],
    [
        ["detectChanges() sur admin dashboard", "Spinner visible malgré les données chargées", "Les rapports et détails s'affichent correctement"],
        ["Sélecteur nav-btn dans le script de capture", "Les onglets Reports n'étaient pas atteints", "Les captures admin nécessaires sont produites"],
        ["Utilisation de localhost pour les captures", "Connexion instable avec 127.0.0.1", "Capture automatisée plus fiable"],
        ["Rebuild complet DOCX/PDF", "PDF resté sur une ancienne version", "Le rapport reflète bien l'annexe étendue"],
    ],
    caption="Tableau 21 : Traçabilité des corrections majeures pendant la réalisation"
)

add_heading("Préparation à la soutenance", level=1, numbered="A.15")
add_paragraph(
    "Pour une présentation orale, la démonstration la plus convaincante consiste à suivre le même chemin "
    "que l'utilisateur réel : connexion, analyse d'un extrait audio, affichage du résultat, puis ouverture "
    "de l'espace administrateur pour montrer la prise en charge d'un signalement. Cette progression donne "
    "du sens à la démonstration et évite de présenter les écrans comme une simple succession de captures."
)
add_paragraph(
    "Il est utile de préparer la soutenance autour de trois idées fortes : la valeur métier du classement "
    "émotionnel, la capacité du système à conserver une traçabilité complète, et la solidité de la chaîne "
    "de génération des artefacts. Ces trois points permettent de justifier à la fois le choix des données, "
    "l'organisation logicielle et l'investissement réalisé dans la documentation."
)
add_table_simple(
    ["Ordre de démo", "Écran / action", "Message à transmettre"],
    [
        ["1", "Accueil et connexion", "Le système contrôle l'accès avant l'analyse"],
        ["2", "Studio d'analyse", "L'utilisateur lance une prédiction sur un audio"],
        ["3", "Résultat et capture", "Le modèle retourne une émotion et un score"],
        ["4", "Signalement", "Le feedback alimente la boucle de validation"],
        ["5", "Admin reports", "L'équipe peut vérifier, retester et documenter"],
        ["6", "Rapport PDF", "Le mémoire rassemble les preuves et les choix techniques"],
    ],
    caption="Tableau 22 : Séquence conseillée pour la démonstration orale"
)

doc.add_page_break()
add_heading("Version mobile de l'application", level=1, numbered="A.16")
add_paragraph(
    "La version mobile reprend exactement le même noyau fonctionnel que l'interface web : authentification, "
    "analyse d'un extrait audio, affichage du résultat de classification, historique des prédictions et "
    "signalement d'une erreur. L'objectif est de proposer une continuité d'usage entre ordinateur et "
    "smartphone, sans changer la logique métier ni la structure des données."
)
add_paragraph(
    "Sur le plan ergonomique, la version mobile privilégie la lisibilité et les gestes courts. Les actions "
    "les plus fréquentes restent accessibles en premier plan, avec une hiérarchie visuelle proche de celle "
    "du site web mais adaptée aux écrans plus compacts. Cette cohérence visuelle facilite la prise en main "
    "et évite à l'utilisateur de réapprendre le fonctionnement du service selon le support utilisé."
)
add_table_simple(
    ["Fonction", "Version web", "Version mobile"],
    [
        ["Connexion", "Formulaire complet sur grand écran", "Formulaire compact et tactile"],
        ["Analyse audio", "Upload, exemples et lecture détaillée", "Actions simplifiées et lecture optimisée"],
        ["Résultat", "Graphiques et probabilités en colonnes", "Blocs empilés pour la lecture verticale"],
        ["Historique", "Profil avec liste d'analyses", "Liste condensée et facile à faire défiler"],
        ["Signalement", "Vue admin détaillée", "Même flux métier, interface adaptée"],
    ],
    caption="Tableau 23 : Correspondance fonctionnelle entre les interfaces web et mobile"
)
add_paragraph(
    "Les captures d'écran mobiles pourront être déposées dans le dossier capturephone/ puis réintégrées "
    "automatiquement dans le rapport lors de la prochaine génération. Le générateur parcourt ce dossier "
    "quand il contient des images et les ajoute à la suite de cette section, ce qui permet de garder le "
    "document synchronisé avec les visuels réellement disponibles."
)
add_directory_figures(CAPTUREPHONE_DIR, "Interface mobile", 26, width=Inches(4.9))

add_heading("Environnement matériel et outils de développement", level=1, numbered="A.17")
add_paragraph(
    "Le développement, les essais et la production des artefacts de ce projet ont été réalisés sur un PC "
    "équipé d'un processeur Intel Core i5 de 12e génération, de 16 Go de RAM et d'une carte graphique RTX "
    "2050. Cette configuration est suffisante pour l'entraînement léger, l'exécution des tests, la capture "
    "des écrans et la génération du rapport complet sans dépendre d'une infrastructure externe."
)
add_paragraph(
    "L'environnement logiciel s'appuie sur Visual Studio Code comme éditeur principal, Python pour la "
    "partie machine learning et l'automatisation du rapport, Node.js et Angular pour le frontend, FastAPI "
    "pour le backend, ainsi que Playwright, python-docx, docx2pdf et PyPDF2 pour les tâches de validation, "
    "de mise en forme et de contrôle du livrable final."
)
add_paragraph(
    "Ces outils ont été choisis pour leur compatibilité avec le projet et pour leur capacité à rendre le "
    "travail reproductible. VS Code facilite la navigation et le débogage, Python centralise le traitement "
    "des données et l'orchestration, tandis que les outils d'automatisation permettent de reconstruire en "
    "quelques commandes les figures, les captures et le mémoire final."
)
add_table_simple(
    ["Élément", "Détail", "Rôle dans le projet"],
    [
        ["PC de développement", "Intel Core i5 12e génération / 16 Go RAM / RTX 2050", "Développement et exécution locale"],
        ["Éditeur", "Visual Studio Code", "Écriture, navigation et débogage"],
        ["Langage principal", "Python", "Backend, ML et génération du rapport"],
        ["Frontend", "Angular / TypeScript", "Interface web et expérience utilisateur"],
        ["Backend", "FastAPI", "API, authentification et signalements"],
        ["Automatisation", "Playwright, python-docx, docx2pdf, PyPDF2", "Captures, export et contrôle du PDF"],
    ],
    caption="Tableau 24 : Matériel et outils de développement utilisés pour le projet"
)
add_paragraph(
    "Au final, la combinaison entre le matériel local, la pile logicielle choisie et la chaîne de génération "
    "automatique garantit un livrable cohérent. Le rapport peut être régénéré à partir du code source et des "
    "captures disponibles, ce qui facilite la maintenance du projet et la préparation de la soutenance."
)

# ─── Sauvegarde ───────────────────────────────────────────────────────────────
output_path = os.path.join(BASE_PATH, "Rapport_PFE_Raddaoui_Wael_Master_DS_IA.docx")
fallback_output_path = os.path.join(BASE_PATH, "Rapport_PFE_Raddaoui_Wael_Master_DS_IA_mobile.docx")
try:
    doc.save(output_path)
except PermissionError:
    output_path = fallback_output_path
    doc.save(output_path)
    print(f"Rapport genere avec succes : {output_path} (fichier principal verrouille)")
else:
    print(f"Rapport genere avec succes : {output_path}")
