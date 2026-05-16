# Seuils de distance (en mètres)
SEUILS = {
    "DANGER":    0.5,
    "ATTENTION": 1.0,
    "PROCHE":    2.0,
}

def get_danger_level(distance: float) -> str:
    """
    Retourne le niveau de danger selon la distance.
    """
    if distance < SEUILS["DANGER"]:
        return "DANGER"
    elif distance < SEUILS["ATTENTION"]:
        return "ATTENTION"
    elif distance < SEUILS["PROCHE"]:
        return "PROCHE"
    return "OK"

def get_voice_message(label_traduit: str, distance: float, danger: str, lang: str = "fr") -> str:
    """
    Génère le message vocal selon la langue et le niveau de danger.
    lang = 'fr' → français
    lang = 'tn' → dialecte tunisien
    """
    if lang == "tn":
        return _message_tn(label_traduit, distance, danger)
    return _message_fr(label_traduit, distance, danger)


# Messages en français
def _message_fr(label: str, distance: float, danger: str) -> str:
    if danger == "DANGER":
        return f"DANGER ! {label} très proche !"
    elif danger == "ATTENTION":
        return f"Attention ! {label} à {distance} mètres"
    elif danger == "PROCHE":
        return f"{label} détecté à {distance} mètres"
    return f"{label} à {distance} mètres"


# Messages en dialecte tunisien
def _message_tn(label: str, distance: float, danger: str) -> str:
    if danger == "DANGER":
        return f"خطر ! {label} قريب بزاف !"
    elif danger == "ATTENTION":
        return f"انتبه ! {label} على بعد {distance} متر"
    elif danger == "PROCHE":
        return f"لقينا {label} على بعد {distance} متر"
    return f"{label} على بعد {distance} متر"
