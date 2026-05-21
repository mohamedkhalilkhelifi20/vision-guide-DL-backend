"""
Script de téléchargement des modèles ML
========================================
Utilise gdown pour Google Drive (gère la confirmation anti-virus automatiquement).
Compatible gdown 6.0.0+
"""

import os
import gdown

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

def _extract_id(url: str) -> str:
    """Extrait le FILE_ID depuis une URL Google Drive."""
    if "/file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    return url


MODELS = {
    "yolov8n.pt": {
        "type": "direct",
        "url": os.environ.get(
            "MODEL_YOLO_URL",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        )
    },
    "mobilenet_fairface.pth": {
        "type": "gdrive",
        "url": os.environ.get(
            "MODEL_CNN_PERSON_URL",
            "https://drive.google.com/file/d/12iyiWnD-6k120jBezuOZqAWm1zMdiUoH/view?usp=drive_link"
        )
    },
    "yolov8n_stairs.pt": {
        "type": "gdrive",
        "url": os.environ.get(
            "MODEL_YOLO_STAIRS_URL",
            "https://drive.google.com/file/d/1Z4cNAu4ku-_RPK7HL5nCs6Ih-gYeMtBU/view?usp=sharing"
        )
    },
    "convnext_stairs.pth": {
        "type": "gdrive",
        "url": os.environ.get(
            "MODEL_CNN_STAIRS_URL",
            "https://drive.google.com/file/d/1wbcLE6b1Ws2vWabDutUXTKhhgwSRa-pQ/view?usp=drive_link"
        )
    },
}

MODEL_DIR = "models"
MIN_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB minimum


def download_gdrive(file_id: str, dest: str) -> bool:
    """Télécharge depuis Google Drive avec gdown 6.0.0+ (sans fuzzy)."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        # ✅ gdown 6.0.0 — plus de paramètre 'fuzzy'
        gdown.download(url, dest, quiet=False)

        if not os.path.exists(dest):
            print(f"   Fichier non créé après téléchargement")
            return False

        size_mb = os.path.getsize(dest) / (1024 * 1024)
        if os.path.getsize(dest) < MIN_SIZE_BYTES:
            os.remove(dest)
            print(f"   Fichier trop petit ({size_mb:.2f} MB) — page HTML Google")
            print(f"     → Vérifiez que le fichier est partagé publiquement.")
            return False

        print(f"   {os.path.basename(dest)} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"   Erreur gdown : {e}")
        return False


def download_direct(url: str, dest: str) -> bool:
    """Télécharge depuis une URL directe (GitHub, etc.)."""
    import requests
    try:
        print(f"  → URL directe : {url[:70]}...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = downloaded / (1024 * 1024)
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

    for filename, config in MODELS.items():
        dest = os.path.join(MODEL_DIR, filename)

        # Déjà présent et valide → skip
        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            if os.path.getsize(dest) < MIN_SIZE_BYTES:
                print(f"    {filename} corrompu ({size_mb:.2f} MB) — re-téléchargement")
                os.remove(dest)
            else:
                print(f"  ⏭  {filename} déjà présent ({size_mb:.1f} MB) — skip")
                continue

        url = config["url"]
        if not url:
            print(f"    {filename} — URL non configurée")
            continue

        print(f"\n   {filename}")

        if config["type"] == "gdrive":
            file_id = _extract_id(url)
            print(f"  → Google Drive ID : {file_id}")
            download_gdrive(file_id, dest)
        else:
            download_direct(url, dest)

    print(f"\n{'='*50}")
    print("  Téléchargement terminé")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()