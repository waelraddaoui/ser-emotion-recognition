# Backend

FastAPI backend with JWT auth, roles (USER/ADMIN), and SER inference.

## Setup

1) Create a virtual env and install deps from the root requirements.
2) Copy `.env.example` to `.env` and edit values.
3) Generate a long random `SECRET_KEY` and set a strong admin password.
4) Start the server:

```bash
uvicorn app.main:app --reload --port 8001
```

## Notes

- Admin user is created from `ADMIN_EMAIL` and `ADMIN_PASSWORD` on startup.
- Audio uploads are stored under `UPLOAD_DIR`.
- Examples are loaded from `DATA_PATH` on startup.
- When `ENVIRONMENT` is not `dev`, `development`, or `local`, unsafe default secrets/admin credentials are rejected on startup.
- Admin-confirmed report corrections can be used as supervised feedback. The backend stores each correction and fine-tunes a feedback model at `FEEDBACK_MODEL_PATH`, while preserving the original baseline model.
