from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from schemas.schemas import DetectResponse
from services.yolo_service import analyze_image
from core.config import settings

router = APIRouter()

@router.post(
    "/detect",
    response_model=DetectResponse,
    summary="Détecter et classifier les objets dans une image",
    tags=["Détection"]
)
async def detect(
        file: UploadFile = File(..., description="Image JPG/PNG depuis la caméra"),
        lang: str = Query(default="fr", description="Langue des messages : 'fr' ou 'tn'")
):
    # Vérifier le format
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=415,
            detail=f"Format non supporté : {file.content_type}. Utilise JPG ou PNG."
        )

    # Lire les bytes
    image_bytes = await file.read()

    # Vérifier la taille (max 10 MB)
    if len(image_bytes) > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image trop grande (max 10 MB)"
        )

    # Analyser — YOLO + CNN + distance + danger
    result = analyze_image(image_bytes, lang=lang)

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Erreur interne")
        )

    return result
