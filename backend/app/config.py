from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "SER Auth Backend"
    environment: str = "dev"

    database_url: str = ""
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    admin_email: str = "admin@example.com"
    admin_password: str = "admin123"

    data_path: str = "../data"
    model_path: str = "../models/best_model_gpu.pth"
    feedback_model_path: str = "../models/best_model_gpu_feedback.pth"
    norm_mean_path: str = "../models/norm_mean.npy"
    norm_std_path: str = "../models/norm_std.npy"
    upload_dir: str = "./uploads"

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", protected_namespaces=())

    @model_validator(mode="after")
    def validate_security_defaults(self):
        if self.environment.lower() not in {"dev", "development", "local"}:
            unsafe_values = {
                "SECRET_KEY": self.secret_key in {"", "change-me", "change-me-now"},
                "ADMIN_EMAIL": self.admin_email == "admin@example.com",
                "ADMIN_PASSWORD": self.admin_password in {"", "admin123"},
            }
            unsafe_names = [name for name, is_unsafe in unsafe_values.items() if is_unsafe]
            if unsafe_names:
                joined = ", ".join(unsafe_names)
                raise ValueError(f"Unsafe production configuration: update {joined}")
        return self

    def resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return (BASE_DIR / p).resolve()

    def resolved_database_url(self) -> str:
        if self.database_url:
            url = self.database_url
            if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
                rel = url.removeprefix("sqlite:///")
                if rel not in (":memory:", "") and not Path(rel).is_absolute():
                    return f"sqlite:///{(BASE_DIR / rel).resolve().as_posix()}"
            return url
        return f"sqlite:///{(BASE_DIR / 'app.db').resolve().as_posix()}"


settings = Settings()
