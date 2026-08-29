import sys
import warnings
warnings.filterwarnings('ignore')

print("1. Starting")
sys.stdout.flush()

print("2. Importing tensorflow FIRST...")
sys.stdout.flush()
import tensorflow as tf
print("3. TensorFlow imported OK")
sys.stdout.flush()

print("4. Importing sklearn...")
sys.stdout.flush()
from sklearn.model_selection import train_test_split
print("5. Sklearn imported OK")
sys.stdout.flush()

print("6. SUCCESS - All imports work when TF is imported first!")
