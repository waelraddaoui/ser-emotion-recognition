import sys
import warnings
warnings.filterwarnings('ignore')

print("1. Starting")
sys.stdout.flush()

print("2. Importing sklearn...")
sys.stdout.flush()
from sklearn.model_selection import train_test_split
print("3. Sklearn imported OK")
sys.stdout.flush()

print("4. Importing tensorflow...")
sys.stdout.flush()
import tensorflow as tf
print("5. TensorFlow imported OK")
sys.stdout.flush()
