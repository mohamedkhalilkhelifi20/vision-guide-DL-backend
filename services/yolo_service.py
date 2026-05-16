from ultralytics import YOLO
from .detector import detect_objects
from .distance import calculate_distance
from .danger import get_danger_level, get_voice_message
from .translator import translate
from .cnn_classifier import cnn_refine
from datetime import datetime

# ── Modèles YOLO ──────────────────────────────────────────────────────────────
# Initialisés à None — chargés une seule fois via load_model() au startup

_model_coco   = None   # yolov8n.pt   — 80 classes COCO (person, car, chair...)
_model_stairs = None   # yolov8n_stairs.pt — fine-tuné pour détecter les escaliers

MODEL_COCO_PATH   = "models/yolov8n.pt"
MODEL_STAIRS_PATH = "models/yolov8n_stairs.pt"


def load_model() -> None:
    """
    Charge les deux modèles YOLO au démarrage.
    - model_coco   : détection générale 80 classes COCO
    - model_stairs : détection spécialisée escaliers (fine-tuné)
    Idempotent — ne recharge pas si déjà en mémoire.
    """
    global _model_coco, _model_stairs

    if _model_coco is None:
        _model_coco = YOLO(MODEL_COCO_PATH)
        print(f"[YOLO]  COCO chargé      : {MODEL_COCO_PATH}")
    else:
        print("[YOLO] COCO déjà chargé — skip")

    if _model_stairs is None:
        try:
            _model_stairs = YOLO(MODEL_STAIRS_PATH)
            print(f"[YOLO]  Stairs chargé   : {MODEL_STAIRS_PATH}")
        except Exception as e:
            print(f"[YOLO]  Stairs non disponible : {e}")
            _model_stairs = None
    else:
        print("[YOLO] Stairs déjà chargé — skip")


def analyze_image(image_bytes: bytes, lang: str = "fr") -> dict:

    if _model_coco is None:
        return {
            "success":    False,
            "count":      0,
            "detections": [],
            "error":      "Modèle YOLO non initialisé — appeler load_model() au startup"
        }

    try:
        # ── Étape 1 : Détection YOLO COCO (80 classes générales) ─────────────
        raw_detections, image_height = detect_objects(_model_coco, image_bytes)

        # ── Étape 2 : Détection YOLO Stairs (si modèle disponible) ───────────
        # On fusionne les détections stairs avec les détections COCO
        if _model_stairs is not None:
            stairs_detections, _ = detect_objects(_model_stairs, image_bytes)
            # Le modèle stairs retourne le label "stairs" — on l'ajoute directement
            raw_detections += stairs_detections

        if not raw_detections:
            return {
                "success":    True,
                "count":      0,
                "detections": []
            }

        # ── Étape 3 : Enrichir chaque détection ──────────────────────────────
        detections = []
        for detection in raw_detections:
            label_yolo = detection["label"]
            bbox       = detection["bbox"]
            bbox_h     = detection["bbox_height"]
            conf       = detection["confidence"]

            # CNN affine le label si disponible pour ce label
            # "person"  → MobileNetV2  → enfant / adulte / personne_agee
            # "stairs"  → ConvNeXt     → pas_stairs / stairs (confirmation)
            label_final   = cnn_refine(label_yolo, image_bytes, bbox)

            # Si CNN dit "pas_stairs" → on ignore cette détection
            if label_final == "pas_stairs":
                print(f"[YOLO] Fausse détection stairs ignorée (CNN: pas_stairs)")
                continue

            distance      = calculate_distance(label_final, bbox_h, image_height)
            danger        = get_danger_level(distance)
            label_traduit = translate(label_final, lang=lang)
            message       = get_voice_message(label_traduit, distance, danger, lang=lang)

            detections.append({
                "label":           label_yolo,
                "label_refined":   label_final,
                "label_fr":        label_traduit,
                "detected_at":     datetime.now().isoformat(),
                "confidence":      conf,
                "distance_meters": distance,
                "danger_level":    danger,
                "voice_message":   message,
            })

        # ── Étape 4 : Trier par niveau de danger ─────────────────────────────
        ordre = {"DANGER": 0, "ATTENTION": 1, "PROCHE": 2, "OK": 3}
        detections.sort(key=lambda x: ordre[x["danger_level"]])

        return {
            "success":    True,
            "count":      len(detections),
            "detections": detections
        }

    except Exception as e:
        return {
            "success":    False,
            "count":      0,
            "detections": [],
            "error":      str(e)
        }
