import json
import os

# Charger objects_db.json — chemin relatif depuis services/ vers la racine du projet
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "objects_db.json")
with open(DB_PATH, "r", encoding="utf-8") as f:
    OBJECTS_DB = json.load(f)

MIN_BBOX       = 5.0
HAUTEUR_DEFAUT = 0.50


def get_focale(image_height: int) -> float:
    return image_height * 0.8


def calculate_distance(label: str, bbox_height: float, image_height: int) -> float:
    """
    Distance (m) = (Hauteur_Réelle × Focale) / Hauteur_BBox_pixels
    """
    if bbox_height < MIN_BBOX:
        return 99.9

    hauteur = OBJECTS_DB.get(label, {}).get("hauteur", HAUTEUR_DEFAUT)
    focale  = get_focale(image_height)

    return round((hauteur * focale) / bbox_height, 1)
