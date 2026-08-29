import os
import shutil
import uuid
import importlib
import logging
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session, joinedload

from .config import settings
import re
from .database import Base, engine, get_db, SessionLocal
from . import models, schemas, auth

logger = logging.getLogger(__name__)

security = HTTPBearer()

app = FastAPI(title="SER Auth Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXAMPLES_CACHE: List[dict] = []

EMOTION_NAMES = ("Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised")
CURATED_DATASET_DIRNAME = "admin_curated"
PROJECT_ROOT = settings.resolve_path("..")
TRAINING_LOG_PATH = PROJECT_ROOT / "logs" / "admin_training.log"
_TRAINING_LOCK = threading.Lock()
_TRAINING_JOB = {
    "running": False,
    "status": "IDLE",
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "log_path": str(TRAINING_LOG_PATH),
    "error_message": None,
}

_AUDIO_MIME_EXT = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/vorbis": ".ogg",
    "audio/aac": ".aac",
    "audio/x-m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/webm": ".webm",
}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
_AUDIO_MEDIA = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}
_EMOTION_EMOJI = {
    "Neutral": "😐",
    "Calm": "😌",
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fearful": "😨",
    "Disgust": "🤢",
    "Surprised": "😲",
}

_ML_MODULE = None
_ML_IMPORT_ERROR: Exception | None = None


def _get_ml():
    global _ML_MODULE, _ML_IMPORT_ERROR
    if _ML_MODULE is not None or _ML_IMPORT_ERROR is not None:
        return _ML_MODULE
    try:
        _ML_MODULE = importlib.import_module(".ml", __package__)
    except Exception as exc:
        _ML_IMPORT_ERROR = exc
        _ML_MODULE = None
    return _ML_MODULE


def _require_ml():
    ml_module = _get_ml()
    if ml_module is None:
        detail = "ML dependencies not installed on the server"
        if _ML_IMPORT_ERROR is not None:
            detail = f"ML import failed: {_ML_IMPORT_ERROR}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )
    return ml_module


def _ensure_upload_dir() -> Path:
    upload_dir = settings.resolve_path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _ensure_curated_dataset_dirs() -> Path:
    root = settings.resolve_path(settings.data_path) / CURATED_DATASET_DIRNAME
    for emotion in EMOTION_NAMES:
        (root / emotion).mkdir(parents=True, exist_ok=True)
    return root


def _curated_dataset_counts(root: Path | None = None) -> dict[str, int]:
    root = root or _ensure_curated_dataset_dirs()
    counts = {}
    for emotion in EMOTION_NAMES:
        emotion_dir = root / emotion
        counts[emotion] = sum(
            1
            for item in emotion_dir.iterdir()
            if item.is_file() and item.suffix.lower() in _AUDIO_EXTS
        )
    return counts


def _read_training_log_tail(max_chars: int = 5000) -> str:
    if not TRAINING_LOG_PATH.exists():
        return ""
    try:
        text = TRAINING_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _training_status_out() -> schemas.TrainingJobStatus:
    with _TRAINING_LOCK:
        snapshot = dict(_TRAINING_JOB)

    return schemas.TrainingJobStatus(
        running=bool(snapshot["running"]),
        status=str(snapshot["status"]),
        started_at=snapshot["started_at"],
        finished_at=snapshot["finished_at"],
        return_code=snapshot["return_code"],
        log_path=snapshot["log_path"],
        log_tail=_read_training_log_tail(),
        error_message=snapshot["error_message"],
    )


def _run_admin_training_job(new_only: bool = True) -> None:
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepare_flag = "--new-only" if new_only else "--admin-only"
    admin_features = "X_admin_feedback_cnn.npy"
    admin_labels = "y_admin_feedback_labels.npy"
    baseline_model_path = settings.resolve_path(settings.model_path)
    active_model_path = settings.resolve_path(settings.feedback_model_path)
    initial_model_path = active_model_path if active_model_path.exists() else baseline_model_path
    commands = [
        [sys.executable, "-B", "prepare_data.py", prepare_flag, "--output-prefix", "admin_feedback"],
        [
            sys.executable,
            "-B",
            "train_model_gpu.py",
            "--features",
            admin_features,
            "--labels",
            admin_labels,
            "--model-output",
            str(active_model_path),
            "--initial-model",
            str(initial_model_path),
        ],
    ]

    return_code = 0
    error_message = None
    try:
        with TRAINING_LOG_PATH.open("w", encoding="utf-8", errors="replace") as log:
            log.write(f"Admin training started at {datetime.now(timezone.utc).isoformat()}\n")
            log.write(f"Python: {sys.executable}\n\n")
            log.flush()

            skip_rest = False
            for command in commands:
                log.write(f"\n=== Running: {' '.join(command)} ===\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                return_code = process.wait()
                log.write(f"\n=== Exit code: {return_code} ===\n")
                log.flush()
                # If prepare_data printed marker saying there are no new curated files,
                # treat as successful but skip the training step.
                try:
                    log_text = TRAINING_LOG_PATH.read_text(encoding='utf-8', errors='replace')
                except Exception:
                    log_text = ''
                if '=== PREPARE_DATA: NO_NEW_CURATED_FILES ===' in log_text and 'prepare_data.py' in ' '.join(command):
                    log.write("\nNo new curated files detected; skipping model training.\n")
                    log.flush()
                    skip_rest = True
                    return_code = 0
                    break

                if return_code != 0:
                    error_message = f"Training command failed: {' '.join(command)}"
                    break

            if return_code == 0:
                if skip_rest:
                    log.write("\nAdmin training skipped because there were no new curated files.\n")
                else:
                    log.write(f"\nTrained feedback model saved at: {active_model_path}\n")
                    log.write("\nTraining completed successfully. Reloading model on next request.\n")
                    ml_module = _get_ml()
                    if ml_module is not None:
                        ml_module._model = None
                        ml_module._norm_mean = None
                        ml_module._norm_std = None
                        ml_module.load_model()
    except Exception as exc:
        logger.error("Admin training job failed", exc_info=True)
        return_code = 1
        error_message = str(exc)
    finally:
        with _TRAINING_LOCK:
            _TRAINING_JOB["running"] = False
            _TRAINING_JOB["status"] = "COMPLETED" if return_code == 0 else "FAILED"
            _TRAINING_JOB["finished_at"] = datetime.now(timezone.utc)
            _TRAINING_JOB["return_code"] = return_code
            _TRAINING_JOB["error_message"] = error_message


def _resolve_audio_extension(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext:
        return ext
    if file.content_type:
        return _AUDIO_MIME_EXT.get(file.content_type, ".audio")
    return ".audio"


def _seed_admin(db: Session) -> None:
    admin = db.query(models.User).filter(models.User.email == settings.admin_email).first()
    if admin:
        if not auth.verify_password(settings.admin_password, admin.hashed_password):
            admin.hashed_password = auth.get_password_hash(settings.admin_password)
            admin.role = "ADMIN"
            admin.is_active = True
            db.commit()
        return
    admin = models.User(
        email=settings.admin_email,
        hashed_password=auth.get_password_hash(settings.admin_password),
        role="ADMIN",
        is_active=True,
    )
    db.add(admin)
    db.commit()


def _scan_examples(ml_module) -> List[dict]:
    data_path = settings.resolve_path(settings.data_path)
    if not data_path.exists():
        return []

    seen = set()
    examples = []
    for actor_dir in sorted(data_path.iterdir()):
        if not actor_dir.is_dir() or not actor_dir.name.startswith("Actor_"):
            continue
        for wav_path in sorted(actor_dir.glob("*.wav")):
            parts = wav_path.name.split("-")
            if len(parts) < 7:
                continue
            try:
                code = int(parts[2])
            except ValueError:
                continue
            if code < 1 or code > 8:
                continue
            idx = code - 1
            if idx in seen:
                continue
            seen.add(idx)
            examples.append({
                "id": idx,
                "emotion": ml_module.EMOTION_NAMES[idx],
                "emoji": ml_module.EMOTION_EMOJIS[idx],
                "filename": wav_path.name,
                "path": str(wav_path),
            })
            if len(seen) == len(ml_module.EMOTION_NAMES):
                return examples
    return examples


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_upload_dir()
    _ensure_curated_dataset_dirs()
    db = SessionLocal()
    try:
        _seed_admin(db)
    finally:
        db.close()
    global EXAMPLES_CACHE
    ml_module = _get_ml()
    if ml_module:
        ml_module.load_model()
        EXAMPLES_CACHE = _scan_examples(ml_module)
    else:
        EXAMPLES_CACHE = []


def _get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> models.User:
    token = credentials.credentials
    try:
        payload = auth.get_token_payload(token)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User inactive")
    return user


def _require_admin(user: models.User = Depends(_get_current_user)) -> models.User:
    if user.role != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user


def _build_audio_url(audio_id: int) -> str:
    return f"/audio/{audio_id}"


def _emotion_emoji(emotion: str) -> str:
    return _EMOTION_EMOJI.get(emotion, "🎙️")


def _audio_media_type(path: str) -> str:
    return _AUDIO_MEDIA.get(Path(path).suffix.lower(), "application/octet-stream")


def _copy_audio_to_curated_dataset(audio: models.AudioFile, prediction_id: int, emotion: str) -> schemas.CuratedAudioOut:
    corrected_emotion = emotion.strip()
    if corrected_emotion not in EMOTION_NAMES:
        raise HTTPException(status_code=400, detail="Invalid corrected emotion")

    source = Path(audio.stored_path)
    if not source.exists() or not source.is_file():
        raise HTTPException(status_code=404, detail="Stored audio file not found")

    root = _ensure_curated_dataset_dirs()
    emotion_dir = root / corrected_emotion
    ext = source.suffix.lower()
    if ext not in _AUDIO_EXTS:
        ext = Path(audio.original_name or "").suffix.lower()
    if ext not in _AUDIO_EXTS:
        ext = ".wav"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_name = f"prediction_{prediction_id}_audio_{audio.id}_{stamp}_{uuid.uuid4().hex[:8]}{ext}"
    target_path = emotion_dir / target_name
    shutil.copy2(source, target_path)

    counts = _curated_dataset_counts(root)
    return schemas.CuratedAudioOut(
        success=True,
        emotion=corrected_emotion,
        filename=target_name,
        relative_path=str(target_path.relative_to(settings.resolve_path(settings.data_path))),
        count_for_emotion=counts[corrected_emotion],
    )


def _normalize_probabilities(values) -> List[float]:
    if values is None:
        return []
    return [float(p) for p in values]


def _prediction_to_out(item: models.Prediction) -> schemas.PredictionOut:
    audio = item.audio_file
    return schemas.PredictionOut(
        id=item.id,
        emotion=item.emotion,
        emoji=_emotion_emoji(item.emotion),
        confidence=float(item.confidence),
        probabilities=_normalize_probabilities(item.probabilities),
        created_at=item.created_at,
        audio_id=item.audio_id,
        audio_url=_build_audio_url(item.audio_id) if item.audio_id else None,
        original_name=audio.original_name if audio else None,
        source="upload" if item.audio_id else "example",
    )

def _report_to_out(report, reporter_email: str, reviewer_email: str | None = None) -> schemas.PredictionReportOut:
    prediction = report.prediction
    audio = prediction.audio_file
    correction = getattr(report, "correction", None)
    return schemas.PredictionReportOut(
        id=report.id,
        prediction_id=report.prediction_id,
        reason=report.reason,
        comment=report.comment,
        status=report.status,
        admin_review_note=report.admin_review_note,
        reviewed_at=report.reviewed_at,
        created_at=report.created_at,
        reporter_email=reporter_email,
        prediction_emotion=prediction.emotion,
        prediction_confidence=float(prediction.confidence),
        audio_id=prediction.audio_id,
        audio_url=_build_audio_url(prediction.audio_id) if prediction.audio_id else None,
        original_name=audio.original_name if audio else None,
        source="upload" if prediction.audio_id else "example",
        reviewed_by_email=reviewer_email,
        corrected_emotion=correction.corrected_emotion if correction else None,
        learning_status=correction.training_status if correction else None,
        learned_at=correction.learned_at if correction else None,
    )


def _predict_from_path(audio_path: str, ml_module) -> schemas.EmotionResult:
    features, mfcc, y_audio, sr = ml_module.extract_features(audio_path)
    pred_class, probs = ml_module.predict_emotion(features)

    fig_prob = ml_module.plot_probabilities(probs)
    fig_mfcc = ml_module.plot_mfcc_spectrogram(mfcc, sr)
    fig_wave = ml_module.plot_waveform(y_audio, sr)

    emotion = ml_module.EMOTION_NAMES[pred_class]
    return schemas.EmotionResult(
        class_id=pred_class,
        emotion=emotion,
        emoji=ml_module.EMOTION_EMOJIS[pred_class],
        color=ml_module.EMOTION_COLORS[pred_class],
        confidence=float(probs[pred_class]),
        description=ml_module.EMOTION_DESCRIPTIONS[emotion],
        probabilities=[float(p) for p in probs],
        img_prob=ml_module.fig_to_base64(fig_prob),
        img_mfcc=ml_module.fig_to_base64(fig_mfcc),
        img_wave=ml_module.fig_to_base64(fig_wave),
    )


@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": True}


def _parse_metrics_text(text: str) -> dict:
    """Parse simple classification report text and return structured metrics."""
    # Accuracy
    acc = None
    m = re.search(r'Test Accuracy:\s*([0-9]*\.?[0-9]+)\s*%?', text, flags=re.IGNORECASE)
    if m:
        try:
            acc_val = float(m.group(1))
            if acc_val > 1:
                acc = acc_val / 100.0
            else:
                acc = acc_val
        except ValueError:
            acc = None

    # macro averages for precision/recall/f1
    precision = recall = f1 = None
    mm = re.search(r'^\s*macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+\d+', text, flags=re.IGNORECASE | re.MULTILINE)
    if mm:
        try:
            precision = float(mm.group(1))
            recall = float(mm.group(2))
            f1 = float(mm.group(3))
        except ValueError:
            precision = recall = f1 = None

    # per-class f1 scores
    class_f1 = []
    for line in text.splitlines():
        mline = re.match(r'^\s*([A-Za-z]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+\d+', line)
        if mline:
            emotion = mline.group(1).strip()
            try:
                f1_val = float(mline.group(4))
                if f1_val > 1:
                    f1_val = f1_val / 100.0
                class_f1.append({"emotion": emotion, "f1": f1_val})
            except ValueError:
                continue

    return {
        "accuracy": acc if acc is not None else 0.0,
        "precision": precision if precision is not None else 0.0,
        "recall": recall if recall is not None else 0.0,
        "f1": f1 if f1 is not None else 0.0,
        "class_f1": class_f1,
    }


@app.get("/model/metrics")
def get_model_metrics():
    """Return aggregated model evaluation metrics parsed from evaluation report files.

    The endpoint looks for a human-readable evaluation report (examples: figures/evaluation_metrics.txt,
    eval_report_simple.txt) and parses accuracy, macro averages and per-class F1 scores.
    """
    candidates = [
        PROJECT_ROOT / "figures" / "evaluation_metrics.txt",
        PROJECT_ROOT / "eval_report_simple.txt",
        PROJECT_ROOT / "eval_report.txt",
    ]
    found = None
    for p in candidates:
        try:
            if p.exists():
                found = p
                break
        except Exception:
            continue

    if not found:
        raise HTTPException(status_code=404, detail="Model metrics not available")

    try:
        text = found.read_text(encoding="utf-8", errors="replace")
        parsed = _parse_metrics_text(text)
        return parsed
    except Exception as exc:
        logger.exception("Failed to parse metrics file")
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {exc}")


@app.post("/auth/register", response_model=schemas.TokenPair)
def register(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    exists = db.query(models.User).filter(models.User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        email=payload.email,
        hashed_password=auth.get_password_hash(payload.password),
        role="USER",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    access_token = auth.create_access_token(user.id, user.role)
    refresh_token, jti, expires_at = auth.create_refresh_token(user.id, user.role)
    db.add(models.RefreshToken(user_id=user.id, jti=jti, expires_at=expires_at, revoked=False))
    db.commit()

    return schemas.TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
    )


@app.post("/auth/login", response_model=schemas.TokenPair)
def login(payload: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user or not auth.verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    access_token = auth.create_access_token(user.id, user.role)
    refresh_token, jti, expires_at = auth.create_refresh_token(user.id, user.role)
    db.add(models.RefreshToken(user_id=user.id, jti=jti, expires_at=expires_at, revoked=False))
    db.commit()

    return schemas.TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
    )


@app.post("/auth/refresh", response_model=schemas.TokenPair)
def refresh(payload: schemas.RefreshRequest, db: Session = Depends(get_db)):
    try:
        token_payload = auth.get_token_payload(payload.refresh_token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if token_payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    jti = token_payload.get("jti")
    user_id = token_payload.get("sub")
    if not jti or not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    token_record = db.query(models.RefreshToken).filter(models.RefreshToken.jti == jti).first()
    if not token_record or token_record.revoked:
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    expires_at = token_record.expires_at
    now_utc = datetime.now(timezone.utc)
    if expires_at.tzinfo is None:
        is_expired = expires_at < now_utc.replace(tzinfo=None)
    else:
        is_expired = expires_at < now_utc

    if is_expired:
        token_record.revoked = True
        db.commit()
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user = db.query(models.User).filter(models.User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive")

    token_record.revoked = True
    access_token = auth.create_access_token(user.id, user.role)
    refresh_token, new_jti, expires_at = auth.create_refresh_token(user.id, user.role)
    db.add(models.RefreshToken(user_id=user.id, jti=new_jti, expires_at=expires_at, revoked=False))
    db.commit()

    return schemas.TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user,
    )


@app.post("/auth/logout")
def logout(payload: schemas.RefreshRequest, db: Session = Depends(get_db)):
    try:
        token_payload = auth.get_token_payload(payload.refresh_token)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    jti = token_payload.get("jti")
    if not jti:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    token_record = db.query(models.RefreshToken).filter(models.RefreshToken.jti == jti).first()
    if token_record:
        token_record.revoked = True
        db.commit()

    return {"success": True}


@app.get("/me", response_model=schemas.UserOut)
def get_me(user: models.User = Depends(_get_current_user)):
    return user


@app.get("/me/predictions", response_model=schemas.PredictionsList)
def my_predictions(user: models.User = Depends(_get_current_user), db: Session = Depends(get_db)):
    items = (
        db.query(models.Prediction)
        .options(joinedload(models.Prediction.audio_file))
        .filter(models.Prediction.user_id == user.id)
        .order_by(models.Prediction.created_at.desc())
        .all()
    )
    return schemas.PredictionsList(items=[_prediction_to_out(item) for item in items])


@app.get("/examples", response_model=List[schemas.ExampleOut])
def get_examples():
    ml_module = _require_ml()
    global EXAMPLES_CACHE
    if not EXAMPLES_CACHE:
        EXAMPLES_CACHE = _scan_examples(ml_module)
    result = []
    for ex in EXAMPLES_CACHE:
        result.append(schemas.ExampleOut(
            id=ex["id"],
            emotion=ex["emotion"],
            emoji=ex["emoji"],
            filename=ex["filename"],
            audio_url=f"/examples/{ex['id']}/audio",
        ))
    return result


@app.get("/examples/{example_id}/audio")
def get_example_audio(example_id: int):
    ml_module = _require_ml()
    global EXAMPLES_CACHE
    if not EXAMPLES_CACHE:
        EXAMPLES_CACHE = _scan_examples(ml_module)
    for ex in EXAMPLES_CACHE:
        if ex["id"] == example_id:
            return FileResponse(ex["path"], media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Example not found")


@app.post("/predict/upload", response_model=schemas.PredictionResponse)
async def predict_upload(
    file: UploadFile = File(...),
    user: models.User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    ml_module = _require_ml()
    content_type = file.content_type or ""
    ext = Path(file.filename or "").suffix.lower()
    if not content_type.startswith("audio/") and ext not in _AUDIO_EXTS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Please upload audio (wav, mp3, flac, m4a, ogg).",
        )
    upload_dir = _ensure_upload_dir() / str(user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = file.filename or "upload.wav"
    ext = Path(original_name).suffix.lower() or _resolve_audio_extension(file)
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = upload_dir / stored_name

    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_stats = os.stat(stored_path)
    logger.debug(f"Saved audio file: path={stored_path}, size={file_stats.st_size} bytes, filename={original_name}")
    
    if file_stats.st_size == 0:
        logger.error(f"WARNING: Uploaded audio file is EMPTY (0 bytes)!")

    audio_record = models.AudioFile(
        user_id=user.id,
        original_name=original_name,
        stored_name=stored_name,
        stored_path=str(stored_path),
    )
    db.add(audio_record)
    db.commit()
    db.refresh(audio_record)

    try:
        result = _predict_from_path(str(stored_path), ml_module)
    except Exception as e:
        logger.error(f"Audio processing failed for {stored_path}: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio decode failed: {str(e)}",
        )

    prediction = models.Prediction(
        user_id=user.id,
        audio_id=audio_record.id,
        emotion=result.emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_name="cnn_pytorch",
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return schemas.PredictionResponse(
        success=True,
        prediction_id=prediction.id,
        audio_id=audio_record.id,
        audio_url=_build_audio_url(audio_record.id),
        original_name=audio_record.original_name,
        result=result,
    )


@app.post("/predictions/{prediction_id}/report", response_model=schemas.PredictionReportOut)
def report_prediction(
    prediction_id: int,
    payload: schemas.PredictionReportCreate,
    user: models.User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    prediction = (
        db.query(models.Prediction)
        .options(joinedload(models.Prediction.audio_file))
        .filter(models.Prediction.id == prediction_id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if prediction.user_id != user.id and user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Not allowed")
    if prediction.audio_id is None:
        raise HTTPException(status_code=400, detail="Only uploaded or recorded audios can be reported")

    existing = (
        db.query(models.PredictionReport)
        .filter(
            models.PredictionReport.prediction_id == prediction_id,
            models.PredictionReport.reporter_user_id == user.id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="You already reported this prediction")

    report = models.PredictionReport(
        prediction_id=prediction_id,
        reporter_user_id=user.id,
        reason=payload.reason.strip(),
        comment=payload.comment.strip() if payload.comment else None,
        status="PENDING",
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return _report_to_out(report, user.email)


@app.post("/predict/example", response_model=schemas.PredictionResponse)
def predict_example(
    payload: schemas.ExampleRequest,
    user: models.User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    ml_module = _require_ml()
    example_path = None
    for ex in EXAMPLES_CACHE:
        if ex["id"] == payload.example_id:
            example_path = ex["path"]
            break
    if not example_path:
        raise HTTPException(status_code=404, detail="Example not found")

    result = _predict_from_path(example_path, ml_module)

    prediction = models.Prediction(
        user_id=user.id,
        audio_id=None,
        emotion=result.emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        model_name="cnn_pytorch",
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return schemas.PredictionResponse(
        success=True,
        prediction_id=prediction.id,
        audio_id=None,
        audio_url=None,
        original_name=None,
        result=result,
    )


@app.get("/audio/{audio_id}")
def get_audio(audio_id: int, user: models.User = Depends(_get_current_user), db: Session = Depends(get_db)):
    audio = db.query(models.AudioFile).filter(models.AudioFile.id == audio_id).first()
    if not audio:
        raise HTTPException(status_code=404, detail="Audio not found")
    if audio.user_id != user.id and user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Not allowed")
    return FileResponse(
        audio.stored_path,
        media_type=_audio_media_type(audio.stored_path),
        filename=audio.original_name,
    )


@app.get("/admin/users", response_model=schemas.AdminUsersList)
def admin_users(
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    items = db.query(models.User).order_by(models.User.created_at.desc()).all()
    return schemas.AdminUsersList(items=items)


# Analytics Endpoints
@app.get("/admin/analytics/dashboard", response_model=schemas.PredictionDashboard)
def admin_analytics_dashboard(
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    """Get prediction dashboard analytics."""
    from sqlalchemy import func
    
    # Total predictions
    total_preds = db.query(func.count(models.Prediction.id)).scalar() or 0
    unique_users = db.query(func.count(func.distinct(models.Prediction.user_id))).scalar() or 0
    
    # Average confidence
    avg_conf = db.query(func.avg(models.Prediction.confidence)).scalar() or 0.0
    
    # Emotion distribution
    emotion_counts = db.query(
        models.Prediction.emotion,
        func.count(models.Prediction.id).label('count')
    ).group_by(models.Prediction.emotion).all()
    
    emotions = []
    for emotion, count in emotion_counts:
        percentage = (count / total_preds * 100) if total_preds > 0 else 0
        emotions.append(schemas.EmotionStats(
            emotion=emotion,
            count=count,
            percentage=round(percentage, 2)
        ))
    
    # Confidence metrics
    conf_stats = db.query(
        func.avg(models.Prediction.confidence).label('avg'),
        func.min(models.Prediction.confidence).label('min'),
        func.max(models.Prediction.confidence).label('max'),
    ).first()
    
    confidence_metrics = schemas.ConfidenceMetrics(
        average=float(conf_stats.avg or 0),
        min=float(conf_stats.min or 0),
        max=float(conf_stats.max or 0),
        std_dev=0.0  # Would need more complex query
    )
    
    return schemas.PredictionDashboard(
        total_predictions=total_preds,
        unique_users=unique_users,
        avg_confidence=round(float(avg_conf), 2),
        emotions=emotions,
        confidence_metrics=confidence_metrics
    )


@app.get("/admin/analytics/low-confidence", response_model=schemas.LowConfidencePredictionsList)
def admin_low_confidence_predictions(
    threshold: float = 0.6,
    limit: int = 50,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    """Get predictions below confidence threshold."""
    predictions = db.query(
        models.Prediction.id,
        models.Prediction.emotion,
        models.Prediction.confidence,
        models.Prediction.created_at,
        models.User.email
    ).join(models.User).filter(
        models.Prediction.confidence < threshold
    ).order_by(
        models.Prediction.created_at.desc()
    ).limit(limit).all()
    
    items = [
        schemas.LowConfidencePrediction(
            id=p.id,
            emotion=p.emotion,
            confidence=float(p.confidence),
            created_at=p.created_at,
            user_email=p.email
        )
        for p in predictions
    ]
    
    return schemas.LowConfidencePredictionsList(
        items=items,
        count=len(items)
    )


@app.get("/admin/analytics/users-stats", response_model=schemas.UserStatsList)
def admin_users_stats(
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    """Get user activity statistics."""
    from sqlalchemy import func
    
    user_stats = db.query(
        models.User.id,
        models.User.email,
        models.User.role,
        models.User.is_active,
        models.User.created_at,
        func.count(models.Prediction.id).label('pred_count'),
        func.max(models.Prediction.created_at).label('last_pred')
    ).outerjoin(models.Prediction).group_by(
        models.User.id
    ).order_by(
        func.count(models.Prediction.id).desc()
    ).all()
    
    items = [
        schemas.UserStats(
            id=u.id,
            email=u.email,
            role=u.role,
            is_active=u.is_active,
            prediction_count=u.pred_count or 0,
            last_prediction=u.last_pred,
            created_at=u.created_at
        )
        for u in user_stats
    ]
    
    return schemas.UserStatsList(items=items)


@app.get("/admin/analytics/predictions-list")
def admin_predictions_list(
    emotion: str = None,
    min_confidence: float = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    """Get filtered predictions for explorer."""
    query = db.query(
        models.Prediction,
        models.User.email
    ).join(models.User).filter(
        models.Prediction.confidence >= min_confidence
    )
    
    if emotion and emotion != "all":
        query = query.filter(models.Prediction.emotion == emotion)
    
    predictions = query.order_by(
        models.Prediction.created_at.desc()
    ).limit(limit).all()
    
    items = [
        {
            "id": p.Prediction.id,
            "emotion": p.Prediction.emotion,
            "confidence": float(p.Prediction.confidence),
            "created_at": p.Prediction.created_at,
            "user_email": p.email,
            "audio_id": p.Prediction.audio_id
        }
        for p in predictions
    ]
    
    return {"items": items, "count": len(items)}


@app.post("/admin/users", response_model=schemas.UserOut)
def admin_create_user(
    payload: schemas.UserAdminCreate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    exists = db.query(models.User).filter(models.User.email == payload.email).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        email=payload.email,
        hashed_password=auth.get_password_hash(payload.password),
        role=payload.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.patch("/admin/users/{user_id}", response_model=schemas.UserOut)
def admin_update_user(
    user_id: int,
    payload: schemas.UserUpdate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.role:
        user.role = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.password:
        user.hashed_password = auth.get_password_hash(payload.password)

    db.commit()
    db.refresh(user)
    return user


@app.delete("/admin/users/{user_id}")
def admin_delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    if admin.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete own account")

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"success": True}


@app.get("/admin/predictions", response_model=schemas.AdminPredictionsList)
def admin_predictions(
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    rows = (
        db.query(models.Prediction)
        .options(joinedload(models.Prediction.audio_file))
        .join(models.User, models.Prediction.user_id == models.User.id)
        .order_by(models.Prediction.created_at.desc())
        .add_columns(models.User)
        .all()
    )
    items = []
    for pred, user in rows:
        out = _prediction_to_out(pred)
        items.append(schemas.AdminPredictionOut(
            id=out.id,
            emotion=out.emotion,
            confidence=out.confidence,
            probabilities=out.probabilities,
            created_at=out.created_at,
            audio_id=out.audio_id,
            audio_url=out.audio_url,
            user_email=user.email,
        ))
    return schemas.AdminPredictionsList(items=items)


@app.get("/admin/curated-dataset", response_model=schemas.CuratedDatasetStats)
def admin_curated_dataset_stats(
    admin: models.User = Depends(_require_admin),
):
    root = _ensure_curated_dataset_dirs()
    counts = _curated_dataset_counts(root)
    return schemas.CuratedDatasetStats(
        root=str(root),
        counts=counts,
        total=sum(counts.values()),
    )


@app.post("/admin/predictions/{prediction_id}/curate", response_model=schemas.CuratedAudioOut)
def curate_prediction_audio(
    prediction_id: int,
    payload: schemas.CuratedAudioCreate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    prediction = (
        db.query(models.Prediction)
        .options(joinedload(models.Prediction.audio_file))
        .filter(models.Prediction.id == prediction_id)
        .first()
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if not prediction.audio_file:
        raise HTTPException(status_code=400, detail="Only uploaded or recorded audios can be curated")

    return _copy_audio_to_curated_dataset(
        prediction.audio_file,
        prediction.id,
        payload.corrected_emotion,
    )


@app.post("/admin/training/start", response_model=schemas.TrainingJobStatus)
def start_admin_training(
    new_only: bool = True,
    admin: models.User = Depends(_require_admin),
):
    _ensure_curated_dataset_dirs()
    with _TRAINING_LOCK:
        if _TRAINING_JOB["running"]:
            raise HTTPException(status_code=409, detail="Training is already running")
        _TRAINING_JOB.update({
            "running": True,
            "status": "RUNNING",
            "started_at": datetime.now(timezone.utc),
            "finished_at": None,
            "return_code": None,
            "log_path": str(TRAINING_LOG_PATH),
            "error_message": None,
        })

    thread = threading.Thread(target=_run_admin_training_job, args=(new_only,), daemon=True)
    thread.start()
    return _training_status_out()


@app.get("/admin/training/status", response_model=schemas.TrainingJobStatus)
def admin_training_status(
    admin: models.User = Depends(_require_admin),
):
    return _training_status_out()


@app.get("/admin/reports", response_model=schemas.PredictionReportList)
def admin_reports(
    status: str | None = None,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    query = (
        db.query(models.PredictionReport, models.User.email.label("reporter_email"))
        .join(models.User, models.PredictionReport.reporter_user_id == models.User.id)
        .options(
            joinedload(models.PredictionReport.prediction).joinedload(models.Prediction.audio_file),
            joinedload(models.PredictionReport.correction),
        )
        .order_by(models.PredictionReport.created_at.desc())
    )
    if status and status != "all":
        query = query.filter(models.PredictionReport.status == status)

    rows = query.all()
    items = []
    for report, reporter_email in rows:
        reviewer_email = None
        if report.reviewed_by_user_id:
            reviewer = db.query(models.User.email).filter(models.User.id == report.reviewed_by_user_id).first()
            reviewer_email = reviewer.email if reviewer else None
        items.append(_report_to_out(report, reporter_email, reviewer_email))
    return schemas.PredictionReportList(items=items, count=len(items))

@app.patch("/admin/reports/{report_id}", response_model=schemas.PredictionReportOut)
def review_report(
    report_id: int,
    payload: schemas.PredictionReportReview,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    report = (
        db.query(models.PredictionReport)
        .options(joinedload(models.PredictionReport.prediction).joinedload(models.Prediction.audio_file))
        .filter(models.PredictionReport.id == report_id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    report.status = payload.status
    report.admin_review_note = payload.admin_review_note
    report.reviewed_by_user_id = admin.id
    report.reviewed_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(report)
    reporter = db.query(models.User.email).filter(models.User.id == report.reporter_user_id).first()
    reviewer = db.query(models.User.email).filter(models.User.id == admin.id).first()
    return _report_to_out(report, reporter.email if reporter else "", reviewer.email if reviewer else None)

@app.post("/admin/reports/{report_id}/learn", response_model=schemas.PredictionCorrectionOut)
def learn_from_report(
    report_id: int,
    payload: schemas.PredictionCorrectionCreate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    report = (
        db.query(models.PredictionReport)
        .options(
            joinedload(models.PredictionReport.prediction).joinedload(models.Prediction.audio_file),
            joinedload(models.PredictionReport.correction),
        )
        .filter(models.PredictionReport.id == report_id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if not report.prediction.audio_file:
        raise HTTPException(status_code=400, detail="Reported prediction has no stored audio")

    ml_module = _require_ml()
    corrected_emotion = payload.corrected_emotion.strip()
    if corrected_emotion not in ml_module.EMOTION_NAMES:
        raise HTTPException(status_code=400, detail="Invalid corrected emotion")

    existing = db.query(models.PredictionCorrection).filter(models.PredictionCorrection.report_id == report.id).first()
    if existing:
        correction = existing
        correction.admin_user_id = admin.id
        correction.corrected_emotion = corrected_emotion
        correction.corrected_class_id = ml_module.EMOTION_NAMES.index(corrected_emotion)
        correction.training_status = "TRAINING"
        correction.error_message = None
    else:
        corrected_class_id = ml_module.EMOTION_NAMES.index(corrected_emotion)
        correction = models.PredictionCorrection(
            report_id=report.id,
            prediction_id=report.prediction_id,
            audio_id=report.prediction.audio_id,
            admin_user_id=admin.id,
            previous_emotion=report.prediction.emotion,
            corrected_emotion=corrected_emotion,
            corrected_class_id=corrected_class_id,
            training_status="TRAINING",
        )
        db.add(correction)

    corrected_class_id = correction.corrected_class_id
    report.status = "CONFIRMED"
    report.reviewed_by_user_id = admin.id
    report.reviewed_at = datetime.now(timezone.utc)
    report.admin_review_note = f"Confirmed by admin. Correct label: {corrected_emotion}."

    db.commit()
    db.refresh(correction)

    try:
        result = ml_module.fine_tune_from_admin_correction(
            report.prediction.audio_file.stored_path,
            corrected_class_id,
            epochs=payload.epochs,
            learning_rate=payload.learning_rate,
        )
        correction.training_status = "LEARNED" if result["achieved_target"] else "SAVED"
        correction.loss_before = result["loss_before"]
        correction.loss_after = result["loss_after"]
        correction.model_path = result["model_path"]
        correction.error_message = None if result["achieved_target"] else (
            f"Fine-tuning saved, but retest still predicts {result['prediction_after']}."
        )
        correction.learned_at = datetime.now(timezone.utc)
    except Exception as exc:
        logger.error("Supervised correction training failed", exc_info=True)
        correction.training_status = "FAILED"
        correction.error_message = str(exc)[:1000]

    db.commit()
    db.refresh(correction)
    if "result" in locals():
        correction.prediction_after = result["prediction_after"]
        correction.achieved_target = result["achieved_target"]
    else:
        correction.prediction_after = None
        correction.achieved_target = False
    return correction


@app.post("/admin/reports/{report_id}/curate", response_model=schemas.CuratedAudioOut)
def curate_report_audio(
    report_id: int,
    payload: schemas.CuratedAudioCreate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    report = (
        db.query(models.PredictionReport)
        .options(joinedload(models.PredictionReport.prediction).joinedload(models.Prediction.audio_file))
        .filter(models.PredictionReport.id == report_id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if not report.prediction.audio_file:
        raise HTTPException(status_code=400, detail="Reported prediction has no stored audio")

    curated = _copy_audio_to_curated_dataset(
        report.prediction.audio_file,
        report.prediction_id,
        payload.corrected_emotion,
    )
    report.status = "CONFIRMED"
    report.reviewed_by_user_id = admin.id
    report.reviewed_at = datetime.now(timezone.utc)
    report.admin_review_note = (
        f"Added to supervised curated dataset as {curated.emotion}."
    )
    db.commit()
    return curated


@app.post("/admin/reports/{report_id}/retest")
def retest_reported_prediction(
    report_id: int,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    report = (
        db.query(models.PredictionReport)
        .options(joinedload(models.PredictionReport.prediction).joinedload(models.Prediction.audio_file))
        .filter(models.PredictionReport.id == report_id)
        .first()
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if not report.prediction.audio_file:
        raise HTTPException(status_code=400, detail="Reported prediction has no stored audio")

    ml_module = _require_ml()
    audio_path = report.prediction.audio_file.stored_path
    result = _predict_from_path(audio_path, ml_module)
    return {
        "report_id": report.id,
        "prediction_id": report.prediction_id,
        "result": result,
    }


@app.delete("/admin/predictions/{prediction_id}")
def admin_delete_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    admin: models.User = Depends(_require_admin),
):
    pred = db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    db.delete(pred)
    db.commit()
    return {"success": True}
