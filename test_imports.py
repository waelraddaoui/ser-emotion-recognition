import os
import sys

# Set environment variables BEFORE any other imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTHONUTF8"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info logs

print("1. os and env vars set")

import pickle
import numpy as np
print("2. numpy/pickle imported")

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
print("3. sklearn imported")

print("About to import tensorflow...")
import tensorflow as tf
print("4. tensorflow imported successfully!")

print("About to import keras.layers...")
from tensorflow.keras import layers, models, regularizers
print("5. keras.models/layers imported")

from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
print("6. keras.callbacks imported")

BASE_PATH    = os.getcwd()
MODELS_DIR   = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
print("7. directories ready")

print("All critical imports successful!")
