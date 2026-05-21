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

def _gdrive_direct(url: str) -> str:
    """
    Convertit une URL Google Drive "view" en URL de téléchargement direct.
    Ex: https://drive.google.com/file/d/FILE_ID/view?...
     →  https://drive.google.com/uc?export=download&confirm=t&id=FILE_ID
    """
    if "drive.google.com/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    return url  # déjà une URL directe (GitHub, etc.)


MODELS = {
    "yolov8n.pt": os.environ.get(
        "MODEL_YOLO_URL",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    ),
    "mobilenet_fairface.pth": os.environ.get(
        "MODEL_CNN_PERSON_URL",
        "https://drive.google.com/file/d/12iyiWnD-6k120jBezuOZqAWm1zMdiUoH/view?usp=drive_link"
    ),
    "yolov8n_stairs.pt": os.environ.get(
        "MODEL_YOLO_STAIRS_URL",
        "https://drive.google.com/file/d/1Z4cNAu4ku-_RPK7HL5nCs6Ih-gYeMtBU/view?usp=sharing"
    ),
    "convnext_stairs.pth": os.environ.get(
        "MODEL_CNN_STAIRS_URL",
        "https://drive.google.com/file/d/1wbcLE6b1Ws2vWabDutUXTKhhgwSRa-pQ/view?usp=drive_link"
    ),
}

MODEL_DIR = "models"

# Taille minimale valide pour un fichier modèle (1 MB)
# Si le fichier téléchargé est plus petit → c'est une page HTML Google, pas un vrai modèle
MIN_SIZE_BYTES = 1 * 1024 * 1024


def download_file(url: str, dest: str) -> bool:
    """Télécharge un fichier avec conversion automatique des URLs Google Drive."""
    try:
        # Convertir l'URL si c'est un lien Google Drive "view"
        direct_url = _gdrive_direct(url)
        print(f"  → URL : {direct_url[:80]}...")

        response = requests.get(direct_url, stream=True, timeout=120)
        response.raise_for_status()

        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = downloaded / (1024 * 1024)

        # Vérification : si le fichier est trop petit → c'est une page HTML (échec silencieux)
        if downloaded < MIN_SIZE_BYTES:
            os.remove(dest)
            print(f"  {os.path.basename(dest)} — fichier trop petit ({size_mb:.2f} MB)")
            print(f"     → Google Drive a renvoyé une page HTML au lieu du modèle.")
            print(f"     → Vérifiez que le fichier est partagé en 'Tout le monde avec le lien'.")
            return False

        print(f"   {os.path.basename(dest)} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"   Erreur : {e}")
        return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print("  Téléchargement des modèles ML")
    print(f"{'='*50}\n")

    for filename, url in MODELS.items():
        dest = os.path.join(MODEL_DIR, filename)

        # Déjà présent → vérifier la taille avant de skip
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            if os.path.getsize(dest) < MIN_SIZE_BYTES:
                print(f"   {filename} présent mais corrompu ({size_mb:.2f} MB) — re-téléchargement")
                os.remove(dest)
            else:
                print(f"  ⏭  {filename} déjà présent ({size_mb:.1f} MB) — skip")
                continue

        # URL vide → skip
        if not url:
            print(f"    {filename} — URL non configurée")
            continue

        download_file(url, dest)

    print(f"\n{'='*50}")
    print("  Téléchargement terminé")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()