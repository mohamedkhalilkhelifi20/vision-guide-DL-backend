import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "objects_db.json")
with open(DB_PATH, "r", encoding="utf-8") as f:
    OBJECTS_DB = json.load(f)


def translate(label: str, lang: str = "fr") -> str:
    return OBJECTS_DB.get(label, {}).get(lang, label)
