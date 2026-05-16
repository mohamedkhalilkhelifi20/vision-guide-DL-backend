import os
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

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")

if _raw_origins == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in _raw_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  #  utiliser la variable, pas hardcodé
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup
@app.on_event("startup")
async def startup():
    load_model()
    load_all_cnn_models()

# Routes
app.include_router(detect_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok", "message": "Object-Finder API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}
