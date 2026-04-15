from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.config import settings
from app.db.init_db import init_db
from app.ml.checkpoint_loader import model_registry

from app.api.routes.health import router as health_router
from app.api.routes.patients import router as patients_router
from app.api.routes.eeg_records import router as eeg_router
from app.api.routes.inference import router as inference_router
from app.api.routes.reports import router as reports_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.auth import router as auth_router
from app.api.routes.settings import router as settings_router

app = FastAPI(title=settings.app_name)

# Serve profile photos
profile_photos_dir = Path("storage/profile_photos")
profile_photos_dir.mkdir(parents=True, exist_ok=True)

app.mount(
    "/static/profile_photos",
    StaticFiles(directory=str(profile_photos_dir)),
    name="profile_photos",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    init_db()
    model_registry.load()


app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(patients_router, prefix=settings.api_prefix)
app.include_router(eeg_router, prefix=settings.api_prefix)
app.include_router(inference_router, prefix=settings.api_prefix)
app.include_router(reports_router, prefix=settings.api_prefix)
app.include_router(dashboard_router, prefix=settings.api_prefix)
app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(settings_router, prefix=settings.api_prefix)