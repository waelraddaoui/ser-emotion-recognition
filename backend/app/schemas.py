from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserAdminCreate(BaseModel):
    email: EmailStr
    password: str
    role: str = "USER"


class UserUpdate(BaseModel):
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class UserOut(BaseModel):
    id: int
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserOut


class ExampleOut(BaseModel):
    id: int
    emotion: str
    emoji: str
    filename: str
    audio_url: str


class ExampleRequest(BaseModel):
    example_id: int


class EmotionResult(BaseModel):
    class_id: int
    emotion: str
    emoji: str
    color: str
    confidence: float
    description: str
    probabilities: List[float]
    img_prob: str
    img_mfcc: str
    img_wave: str


class PredictionResponse(BaseModel):
    success: bool
    prediction_id: int
    audio_id: Optional[int] = None
    audio_url: Optional[str] = None
    original_name: Optional[str] = None
    result: EmotionResult


class PredictionOut(BaseModel):
    id: int
    emotion: str
    emoji: Optional[str] = None
    confidence: float
    probabilities: List[float]
    created_at: datetime
    audio_id: Optional[int] = None
    audio_url: Optional[str] = None
    original_name: Optional[str] = None
    source: Optional[str] = None

    class Config:
        from_attributes = True


class PredictionReportCreate(BaseModel):
    reason: str
    comment: Optional[str] = None


class PredictionReportOut(BaseModel):
    id: int
    prediction_id: int
    reason: str
    comment: Optional[str] = None
    status: str
    admin_review_note: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime
    reporter_email: EmailStr
    prediction_emotion: str
    prediction_confidence: float
    audio_id: Optional[int] = None
    audio_url: Optional[str] = None
    original_name: Optional[str] = None
    source: Optional[str] = None
    reviewed_by_email: Optional[EmailStr] = None
    corrected_emotion: Optional[str] = None
    learning_status: Optional[str] = None
    learned_at: Optional[datetime] = None


class PredictionReportList(BaseModel):
    items: List[PredictionReportOut]
    count: int


class PredictionReportReview(BaseModel):
    status: str
    admin_review_note: Optional[str] = None


class PredictionCorrectionCreate(BaseModel):
    corrected_emotion: str
    epochs: int = 80
    learning_rate: float = 0.001


class CuratedAudioCreate(BaseModel):
    corrected_emotion: str


class CuratedAudioOut(BaseModel):
    success: bool
    emotion: str
    filename: str
    relative_path: str
    count_for_emotion: int


class CuratedDatasetStats(BaseModel):
    root: str
    counts: dict[str, int]
    total: int


class TrainingJobStatus(BaseModel):
    running: bool
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    return_code: Optional[int] = None
    log_path: Optional[str] = None
    log_tail: str = ""
    error_message: Optional[str] = None


class PredictionCorrectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    report_id: int
    prediction_id: int
    previous_emotion: str
    corrected_emotion: str
    corrected_class_id: int
    training_status: str
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None
    prediction_after: Optional[str] = None
    achieved_target: Optional[bool] = None
    created_at: datetime
    learned_at: Optional[datetime] = None



class PredictionsList(BaseModel):
    items: List[PredictionOut]


class AdminPredictionOut(BaseModel):
    id: int
    emotion: str
    confidence: float
    probabilities: List[float]
    created_at: datetime
    audio_id: Optional[int] = None
    audio_url: Optional[str] = None
    user_email: EmailStr


class AdminPredictionsList(BaseModel):
    items: List[AdminPredictionOut]


class AdminUsersList(BaseModel):
    items: List[UserOut]


# Analytics schemas
class EmotionStats(BaseModel):
    emotion: str
    count: int
    percentage: float


class ConfidenceMetrics(BaseModel):
    average: float
    min: float
    max: float
    std_dev: float


class PredictionDashboard(BaseModel):
    total_predictions: int
    unique_users: int
    avg_confidence: float
    emotions: List[EmotionStats]
    confidence_metrics: ConfidenceMetrics


class ModelMetrics(BaseModel):
    emotion: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class ModelPerformance(BaseModel):
    metrics: List[ModelMetrics]
    avg_accuracy: float


class LowConfidencePrediction(BaseModel):
    id: int
    emotion: str
    confidence: float
    created_at: datetime
    user_email: str


class LowConfidencePredictionsList(BaseModel):
    items: List[LowConfidencePrediction]
    count: int


class UserStats(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    prediction_count: int
    last_prediction: Optional[datetime]
    created_at: datetime


class UserStatsList(BaseModel):
    items: List[UserStats]
