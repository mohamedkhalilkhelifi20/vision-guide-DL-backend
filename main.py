from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.detect import router as detect_router
from core.config import settings
from services.yolo_service import load_model
from services.cnn_classifier import load_all_cnn_models

app = FastAPI(
    title="Object-Finder API",
    description="Pipeline YOLO + CNN pour l'aide à la navigation des personnes malvoyantes",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()           # YOLOv8n → _model dans yolo_service
    load_all_cnn_models()  # MobileNetV2 + ConvNeXt → _model_cache dans cnn_classifier

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(detect_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "message": "Object-Finder API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
