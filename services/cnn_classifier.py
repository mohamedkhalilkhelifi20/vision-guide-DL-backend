"""
CNN Classifier — Pipeline Object-Finder
========================================
Rôle 1 — "person"  : MobileNetV2   → enfant / adulte / personne_agee
Rôle 2 — "stairs"  : ConvNeXt-Tiny → pas_stairs / stairs

Logique :
    YOLO détecte "person" → cnn_refine() croppe la bbox → MobileNetV2 classifie
    Si aucun CNN pour ce label → retourne le label YOLO original (pas de crash)
"""

import io
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

CNN_REFINEMENTS = {

    # Rôle 1 — person → enfant / adulte / personne_agee
    # Modèle : MobileNetV2 fine-tuné sur FairFace
    "person": {
        "model_path": "models/mobilenet_fairface.pth",
        "classes": ["adulte", "enfant", "personne_agee"],
        "threshold":  0.60,
        "model_type": "mobilenetv2",
    },

    # Rôle 2 — stairs → pas_stairs / stairs
    # Modèle : ConvNeXt-Tiny fine-tuné sur Open Images
    "stairs": {
        "model_path": "models/convnext_stairs.pth",
        "classes":    ["pas_stairs", "stairs"],
        "threshold":  0.70,
        "model_type": "convnext",
    },
}

# ── TRANSFORMATIONS ───────────────────────────────────────────────────────────
# Normalisation ImageNet standard — identique pour tous les modèles

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )
])

# ── CACHE ─────────────────────────────────────────────────────────────────────
_model_cache: dict = {}


# ── CONSTRUCTION DE L'ARCHITECTURE ───────────────────────────────────────────

def _build_model(model_type: str, num_classes: int) -> nn.Module:
    """
    Reconstruit l'architecture exacte utilisée pendant l'entraînement Kaggle.

    MobileNetV2 — tête custom avec couche cachée (lue depuis l'erreur state_dict) :
        classifier.0 → Dropout(0.2)
        classifier.1 → Linear(1280, 256)   ← couche cachée
        classifier.2 → ReLU()
        classifier.3 → Dropout(0.5)
        classifier.4 → Linear(256, num_classes)   ← sortie

    ConvNeXt-Tiny :
        classifier.0 → LayerNorm
        classifier.1 → Flatten
        classifier.2 → Linear(768, num_classes)
    """
    if model_type == "mobilenetv2":
        base = models.mobilenet_v2(weights=None)
        # Architecture exacte du notebook Kaggle (lue depuis le log) :
        #   classifier.0 → Dropout(0.3)
        #   classifier.1 → Linear(1280, 256)
        #   classifier.2 → ReLU(inplace=True)
        #   classifier.3 → Dropout(0.2)
        #   classifier.4 → Linear(256, num_classes)
        base.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        return base

    elif model_type == "convnext":
        base = models.convnext_tiny(weights=None)
        in_features = base.classifier[2].in_features          # 768
        base.classifier[2] = nn.Linear(in_features, num_classes)
        return base

    else:
        raise ValueError(f"[CNN] model_type inconnu : '{model_type}'")


# ── CHARGEMENT ────────────────────────────────────────────────────────────────

def _load_model(model_path: str, num_classes: int, model_type: str) -> nn.Module:
    """
    Charge les poids fine-tunés dans l'architecture reconstruite.
    Met en cache — chargement unique au premier appel.
    """
    if model_path in _model_cache:
        return _model_cache[model_path]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = _build_model(model_type, num_classes)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    base.load_state_dict(state_dict)
    base.to(device)
    base.eval()

    _model_cache[model_path] = base
    print(f"[CNN] ✅ {model_path} chargé ({num_classes} classes, {model_type}) sur {device}")
    return base


# ── PRÉCHARGEMENT AU STARTUP ──────────────────────────────────────────────────

def load_all_cnn_models() -> None:
    """
    Précharge tous les CNN configurés dans CNN_REFINEMENTS.
    Appelé une seule fois dans main.py au démarrage du serveur.
    Les modèles absents sont ignorés silencieusement (pas de crash).
    """
    for label, config in CNN_REFINEMENTS.items():
        path = config["model_path"]
        if not os.path.exists(path):
            print(f"[CNN] ⚠️  Modèle absent : {path} (label '{label}' ignoré)")
            continue
        try:
            _load_model(path, len(config["classes"]), config["model_type"])
        except Exception as e:
            print(f"[CNN] ❌ Erreur chargement '{label}' : {e}")


# ── CROP DE LA RÉGION YOLO ────────────────────────────────────────────────────

def _crop_region(image_bytes: bytes, bbox: dict, padding: float = 0.10) -> Image.Image:
    """
    Extrait et croppe la région détectée par YOLO.
    Ajoute un padding de 10% pour garder un peu de contexte autour de l'objet.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h  = image.size

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return image.crop((x1, y1, x2, y2))


# ── CLASSIFICATION — FONCTION PRINCIPALE ─────────────────────────────────────

def cnn_refine(yolo_label: str, image_bytes: bytes, bbox: dict) -> str:
    """
    Affine le label YOLO avec le CNN si un modèle est disponible pour ce label.

    Args:
        yolo_label  : label YOLO brut     (ex: "person")
        image_bytes : image complète      (bytes)
        bbox        : {"x1", "y1", "x2", "y2"}

    Returns:
        label affiné (ex: "enfant") ou label YOLO original si pas de CNN
    """
    config = CNN_REFINEMENTS.get(yolo_label)
    if config is None:
        return yolo_label                  # pas de CNN pour ce label → YOLO direct

    try:
        # 1 — Charger le modèle (depuis cache)
        model = _load_model(
            config["model_path"],
            len(config["classes"]),
            config["model_type"]
        )

        # 2 — Cropper la région YOLO
        cropped = _crop_region(image_bytes, bbox)

        # 3 — Préparer le tensor
        device = next(model.parameters()).device
        tensor = TRANSFORM(cropped).unsqueeze(0).to(device)

        # 4 — Inférence
        with torch.no_grad():
            outputs       = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        conf_value      = confidence.item()
        predicted_class = config["classes"][predicted.item()]

        # 5 — Seuil : si confiance trop faible → garder le label YOLO
        if conf_value < config["threshold"]:
            print(f"[CNN] Confiance faible ({conf_value:.2f}) → garde '{yolo_label}'")
            return yolo_label

        print(f"[CNN] {yolo_label} → {predicted_class} ({conf_value:.2f})")
        return predicted_class

    except FileNotFoundError:
        print(f"[CNN] Modèle introuvable : {config['model_path']} → garde '{yolo_label}'")
        return yolo_label

    except Exception as e:
        print(f"[CNN] Erreur pour '{yolo_label}' : {e} → garde label YOLO")
        return yolo_label
