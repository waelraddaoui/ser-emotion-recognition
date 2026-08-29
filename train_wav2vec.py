import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2Processor, 
    TrainingArguments, 
    Trainer
)

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, "data")
MODELS_DIR = os.path.join(BASE_PATH, "models", "wav2vec2-ser")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAME = "facebook/wav2vec2-base-960h"
N_LABELS = 8  # Our dataset has 8 emotions (Neutral to Surprised)
TARGET_SR = 16000

# ─── Data Preparation ─────────────────────────────────────────────────────────
def load_dataset():
    data = []
    for actor_dir in sorted(os.listdir(DATA_PATH)):
        actor_path = os.path.join(DATA_PATH, actor_dir)
        if os.path.isdir(actor_path) and actor_dir.startswith("Actor_"):
            for fname in os.listdir(actor_path):
                if fname.endswith(".wav"):
                    parts = fname.split("-")
                    if len(parts) >= 7:
                        emotion_code = int(parts[2])
                        if emotion_code <= 8:  # Only use all 8 emotions
                            data.append({
                                "path": os.path.join(actor_path, fname),
                                "label": emotion_code - 1
                            })
    return pd.DataFrame(data)

# ─── Custom Dataset Class ─────────────────────────────────────────────────────
class SERDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav, sr = librosa.load(row["path"], sr=TARGET_SR)
        
        # Max duration 3s for memory efficiency
        max_samples = 3 * TARGET_SR
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        else:
            wav = np.pad(wav, (0, max_samples - len(wav)))

        inputs = self.processor(wav, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        return {
            "input_values": inputs.input_values.squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long)
        }

# ─── Main Training ────────────────────────────────────────────────────────────
def main():
    print("--- Starting Wav2Vec 2.0 SOTA Training ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load data
    df = load_dataset()
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    print(f"Dataset Size: {len(df)} (Train: {len(train_df)}, Val: {len(val_df)})")

    # 2. Processor & Model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=N_LABELS
    ).to(device)

    # Freeze feature encoder
    model.freeze_feature_encoder()

    # 3. Datasets
    train_dataset = SERDataset(train_df, processor)
    val_dataset   = SERDataset(val_df, processor)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=MODELS_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none"
    )

    # 5. Accuracy Metric
    import evaluate
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("Beginning fine-tuning...")
    trainer.train()

    # 8. Save
    print("Saving model...")
    model.save_pretrained(MODELS_DIR)
    processor.save_pretrained(MODELS_DIR)
    print(f"SOTA Model Saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()
