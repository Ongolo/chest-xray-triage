import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm
import io
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(page_title="Chest X-Ray Triage", page_icon="🫁", layout="wide")

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

LABELS_FR = [
    "Atélectasie", "Cardiomégalie", "Épanchement", "Infiltration",
    "Masse", "Nodule", "Pneumonie", "Pneumothorax",
    "Consolidation", "Œdème", "Emphysème", "Fibrose",
    "Épaississement pleural", "Hernie"
]

RECOMMANDATIONS = {
    "Atelectasis": "Encourager des exercices respiratoires profonds. Éviter la sédentarité prolongée.",
    "Cardiomegaly": "Surveiller la pression artérielle. Réduire la consommation de sel et consulter un cardiologue.",
    "Effusion": "Éviter l'exposition au froid. Repos recommandé et suivi médical rapproché.",
    "Infiltration": "Maintenir une bonne hygiène respiratoire. Éviter les environnements poussiéreux.",
    "Mass": "Consultation spécialisée urgente recommandée. Bilan complémentaire nécessaire.",
    "Nodule": "Suivi radiologique régulier recommandé. Éviter le tabac absolument.",
    "Pneumonia": "Repos complet, hydratation abondante. Traitement antibiotique selon prescription médicale.",
    "Pneumothorax": "Consultation urgente. Éviter les efforts physiques intenses.",
    "Consolidation": "Traitement antibiotique recommandé. Repos et suivi médical.",
    "Edema": "Réduire les apports en sel et liquides. Surélever les membres inférieurs au repos.",
    "Emphysema": "Arrêt du tabac impératif. Rééducation respiratoire recommandée.",
    "Fibrosis": "Éviter les irritants respiratoires. Suivi pneumologique régulier.",
    "Pleural Thickening": "Suivi radiologique annuel. Éviter l'exposition à l'amiante.",
    "Hernia": "Éviter le port de charges lourdes. Consultation chirurgicale recommandée."
}

MODEL_PATH = r"C:\Users\ONGOLO YVAN\Desktop\chest-xray-triage\src\models\densenet_best.pt"
AE_PATH    = r"C:\Users\ONGOLO YVAN\Desktop\chest-xray-triage\src\models\autoencoder_best.pt"

# ============================================================
# MODÈLES
# ============================================================
@st.cache_resource
def load_classifier():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 14)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

@st.cache_resource
def load_autoencoder():
    model = Autoencoder()
    model.load_state_dict(torch.load(AE_PATH, map_location='cpu'))
    model.eval()
    return model

# ============================================================
# GÉNÉRATION PDF
# ============================================================
def generate_pdf(filename, probs, anomaly_score, seuil):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Titre
    story.append(Paragraph("RAPPORT D'ANALYSE RADIOLOGIQUE", styles['Title']))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"Fichier : {filename}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))

    # Résultats
    story.append(Paragraph("RÉSULTATS DE L'ANALYSE", styles['Heading2']))
    story.append(Spacer(1, 0.3*cm))

    data = [["Pathologie", "Probabilité", "Statut"]]
    sorted_results = sorted(zip(LABELS_FR, LABELS, probs), key=lambda x: x[2], reverse=True)
    for label_fr, label_en, prob in sorted_results:
        statut = "DÉTECTÉ" if prob > 0.5 else "Normal"
        data.append([label_fr, f"{prob:.1%}", statut])

    table = Table(data, colWidths=[7*cm, 4*cm, 4*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e88e5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4ff')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5*cm))

    # Score anomalie
    story.append(Paragraph("SCORE D'ANOMALIE", styles['Heading2']))
    statut_anomalie = "ATYPIQUE ⚠️" if anomaly_score > seuil else "NORMAL ✅"
    story.append(Paragraph(f"Score : {anomaly_score:.4f} — Statut : {statut_anomalie}", styles['Normal']))
    story.append(Spacer(1, 0.5*cm))

    # Recommandations
    story.append(Paragraph("RECOMMANDATIONS PRÉVENTIVES", styles['Heading2']))
    story.append(Spacer(1, 0.3*cm))
    detected = [(lf, le, p) for lf, le, p in sorted_results if p > 0.5]
    if detected:
        for label_fr, label_en, prob in detected:
            story.append(Paragraph(f"• {label_fr} ({prob:.1%}) :", styles['Heading3']))
            story.append(Paragraph(RECOMMANDATIONS[label_en], styles['Normal']))
            story.append(Spacer(1, 0.2*cm))
    else:
        story.append(Paragraph("Aucune pathologie significative détectée. Maintenir un suivi régulier.", styles['Normal']))

    # Avertissement
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "⚠️ AVERTISSEMENT : Ce rapport est généré par un prototype éducatif. "
        "Il ne remplace en aucun cas l'avis d'un professionnel de santé qualifié.",
        styles['Normal']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================
# INTERFACE
# ============================================================
st.markdown("""
    <h1 style='text-align: center; color: #1e88e5;'>
        🫁 Système d'aide au tri radiologique
    </h1>
    <p style='text-align: center; color: gray;'>
        Analyse automatique de radiographies thoraciques par Deep Learning
    </p>
    <hr>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ℹ️ À propos")
    st.write("DenseNet121 entraîné sur ChestMNIST+ pour détecter 14 pathologies thoraciques.")
    st.markdown("### 📊 Performances")
    st.metric("AUC DenseNet", "0.6568")
    st.metric("AUC ViT", "0.6546")
    st.metric("AUC CNN Scratch", "0.6108")
    st.markdown("### ⚠️ Avertissement")
    st.warning("Prototype éducatif uniquement. Ne pas utiliser pour des décisions cliniques.")

uploaded_file = st.file_uploader("📂 Charger une radiographie thoracique", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')