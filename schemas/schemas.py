from pydantic import BaseModel


# OopCompanion:suppressRename


# ── DETECTION ──────────────────────────────────────────────

class DetectionItem(BaseModel):
    label:           str
    label_refined:   str
    label_fr:        str
    confidence:      float
    distance_meters: float
    danger_level:    str
    voice_message:   str
    detected_at:     str

    model_config = {
        "json_schema_extra": {
            "example": {
                "label":           "person",
                "label_refined":   "enfant",
                "label_fr":        "enfant",
                "confidence":      0.95,
                "distance_meters": 0.4,
                "danger_level":    "DANGER",
                "voice_message":   "DANGER ! enfant très proche !",
                "detected_at":     "2024-01-15T10:30:00"
            }
        }
    }


class DetectResponse(BaseModel):
    success:    bool
    count:      int
    detections: list[DetectionItem]

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "count":   1,
                "detections": [
                    {
                        "label":           "person",
                        "label_refined":   "enfant",
                        "label_fr":        "enfant",
                        "confidence":      0.95,
                        "distance_meters": 0.4,
                        "danger_level":    "DANGER",
                        "voice_message":   "DANGER ! enfant très proche !",
                        "detected_at":     "2024-01-15T10:30:00"
                    }
                ]
            }
        }
    }
