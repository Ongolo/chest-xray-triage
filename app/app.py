import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia"
]

MODEL_PATH = r"C:\Users\ONGOLO YVAN\Desktop\chest-xray-triage\src\models\densenet_best.pt"
AE_PATH    = r"C:\Users\ONGOLO YVAN\Desktop\chest-xray-triage\src\models\autoencoder_best.pt"

# ============================================================
# MODÈLES
# ============================================================
import torchvision.models as models

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
# INTERFACE
# ============================================================
st.title("🫁 Système d'aide au tri radiologique")
st.write("Chargez une radiographie thoracique pour obtenir une analyse automatique.")

uploaded_file = st.file_uploader("Choisir une radiographie", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Radiographie chargée", width=300)

    # Preprocessing pour le classifieur
    transform_cls = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5057, 0.5057, 0.5057], std=[0.2896, 0.2896, 0.2896])
    ])

    # Preprocessing pour l'autoencodeur
    transform_ae = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5057], std=[0.2896])
    ])

    img_cls = transform_cls(image).unsqueeze(0)
    img_ae  = transform_ae(image).unsqueeze(0)

    # Prédictions supervisées
    st.subheader("📊 Prédictions des pathologies")
    classifier = load_classifier()
    with torch.no_grad():
        outputs = classifier(img_cls)
        probs = torch.sigmoid(outputs).squeeze().numpy()

    for label, prob in sorted(zip(LABELS, probs), key=lambda x: x[1], reverse=True):
        st.progress(float(prob), text=f"{label}: {prob:.1%}")

    # Score d'anomalie
    st.subheader("🔍 Score d'anomalie")
    ae = load_autoencoder()
    with torch.no_grad():
        reconstructed = ae(img_ae)
        anomaly_score = ((img_ae - reconstructed) ** 2).mean().item()

    seuil = 0.05
    st.metric("Score d'anomalie", f"{anomaly_score:.4f}")
    if anomaly_score > seuil:
        st.error("⚠️ Image atypique détectée — cas inhabituel")
    else:
        st.success("✅ Image dans la distribution normale")