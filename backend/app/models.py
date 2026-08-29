from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="USER", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    audio_files = relationship("AudioFile", back_populates="owner")
    predictions = relationship("Prediction", back_populates="owner")
    refresh_tokens = relationship("RefreshToken", back_populates="owner")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    jti = Column(String(64), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="refresh_tokens")


class AudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    original_name = Column(String(255), nullable=False)
    stored_name = Column(String(255), nullable=False)
    stored_path = Column(String(1024), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="audio_files")
    predictions = relationship("Prediction", back_populates="audio_file")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    audio_id = Column(Integer, ForeignKey("audio_files.id"), nullable=True)
    emotion = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)
    model_name = Column(String(100), nullable=False, default="cnn_pytorch")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    owner = relationship("User", back_populates="predictions")
    audio_file = relationship("AudioFile", back_populates="predictions")


class PredictionReport(Base):
    __tablename__ = "prediction_reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    reporter_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    reason = Column(String(120), nullable=False)
    comment = Column(String(1000), nullable=True)
    status = Column(String(20), default="PENDING", nullable=False)
    admin_review_note = Column(String(1000), nullable=True)
    reviewed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prediction = relationship("Prediction")
    reporter = relationship("User", foreign_keys=[reporter_user_id])
    reviewed_by = relationship("User", foreign_keys=[reviewed_by_user_id])
    correction = relationship("PredictionCorrection", back_populates="report", uselist=False)


class PredictionCorrection(Base):
    __tablename__ = "prediction_corrections"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("prediction_reports.id"), unique=True, nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, index=True)
    audio_id = Column(Integer, ForeignKey("audio_files.id"), nullable=False, index=True)
    admin_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    previous_emotion = Column(String(50), nullable=False)
    corrected_emotion = Column(String(50), nullable=False)
    corrected_class_id = Column(Integer, nullable=False)
    training_status = Column(String(20), default="PENDING", nullable=False)
    loss_before = Column(Float, nullable=True)
    loss_after = Column(Float, nullable=True)
    model_path = Column(String(1024), nullable=True)
    error_message = Column(String(1000), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    learned_at = Column(DateTime(timezone=True), nullable=True)

    report = relationship("PredictionReport", back_populates="correction")
    prediction = relationship("Prediction")
    audio_file = relationship("AudioFile")
    admin = relationship("User", foreign_keys=[admin_user_id])
