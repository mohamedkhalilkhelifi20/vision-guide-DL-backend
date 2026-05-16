"""
Script de téléchargement des modèles ML
========================================
À exécuter avant de lancer le serveur sur Render (Build Command).
Les modèles sont stockés sur Google Drive — liens publics configurés
dans les variables d'environnement Render.

Usage local :
    python download_models.py

Usage Render (Build Command) :
    pip install -r requirements.txt && python download_models.py
"""

import os
import requests

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Les liens Google Drive sont stockés dans les variables d'environnement Render
# Format Google Drive : https://drive.google.com/uc?export=download&id=FILE_ID

MODELS = {
    "yolov8n.pt": os.environ.get(
        "MODEL_YOLO_URL",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        # yolov8n est public sur GitHub — pas besoin de Drive
    ),
    "mobilenet_fairface.pth": os.environ.get(
        "MODEL_CNN_PERSON_URL",
        "https://drive.google.com/file/d/12iyiWnD-6k120jBezuOZqAWm1zMdiUoH/view?usp=drive_link"   # à remplir dans les variables d'environnement Render
    ),
    "yolov8n_stairs.pt": os.environ.get(
        "MODEL_YOLO_STAIRS_URL",
        "https://drive.google.com/file/d/1Z4cNAu4ku-_RPK7HL5nCs6Ih-gYeMtBU/view?usp=sharing"   # à remplir dans les variables d'environnement Render
    ),
    "convnext_stairs.pth": os.environ.get(
        "MODEL_CNN_STAIRS_URL",
        "https://drive.google.com/file/d/1wbcLE6b1Ws2vWabDutUXTKhhgwSRa-pQ/view?usp=drive_link"   # à remplir dans les variables d'environnement Render
    ),
}

MODEL_DIR = "models"


def download_file(url: str, dest: str) -> bool:
    """Télécharge un fichier avec barre de progression simple."""
    try:
        print(f"  → Téléchargement : {url[:60]}...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = downloaded / (1024 * 1024)
        print(f"  ✅ {os.path.basename(dest)} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ❌ Erreur : {e}")
        return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print("  Téléchargement des modèles ML")
    print(f"{'='*50}\n")

    for filename, url in MODELS.items():
        dest = os.path.join(MODEL_DIR, filename)

        # Déjà présent → skip
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  ⏭  {filename} déjà présent ({size_mb:.1f} MB) — skip")
            continue

        # URL vide → skip (modèle optionnel)
        if not url:
            print(f"  ⚠️  {filename} — URL non configurée (variable d'env manquante)")
            continue

        download_file(url, dest)

    print(f"\n{'='*50}")
    print("  Téléchargement terminé")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
