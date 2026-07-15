from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "NeuroXAI Backend"
    api_prefix: str = "/api/v1"
    debug: bool = True

    project_root: Path = Path(__file__).resolve().parents[2]
    storage_dir: Path = project_root / "storage"
    uploads_dir: Path = storage_dir / "uploads"
    artifacts_dir: Path = storage_dir / "artifacts"
    reports_dir: Path = storage_dir / "reports"

    database_url: str = "postgresql+psycopg://postgres:tharu123@localhost:5432/neuroxai"
    checkpoint_path: Path = project_root / "assets" / "checkpoints" / "neuroxai_best_auc.pt"

    # Auth / JWT
    secret_key: str = "change-this-secret-key"
    access_token_expire_minutes: int = 1440

    # NEW: Email verification / SMTP
    frontend_url: str = "http://localhost:5173"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    verification_token_expire_hours: int = 24

    # EEG preprocessing / inference settings
    sfreq_target: int = 100
    l_freq: float = 0.5
    h_freq: float = 30.0
    win_sec: int = 10
    step_sec_seizure: int = 5
    step_sec_nonseizure: int = 10
    default_threshold: float = 0.21
    min_recording_seconds: int = 10

    # Runtime controls for deployment/demo
    inference_batch_size: int = 24
    inference_step_sec: int = 10
    enable_saliency: bool = False

    # Optional demo safety cap for very long EDFs
    max_recording_minutes: int | None = None


settings = Settings()

for p in [
    settings.storage_dir,
    settings.uploads_dir,
    settings.artifacts_dir,
    settings.reports_dir,
]:
    p.mkdir(parents=True, exist_ok=True)