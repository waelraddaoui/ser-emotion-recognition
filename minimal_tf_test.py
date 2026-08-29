import sys
import warnings
warnings.filterwarnings('ignore')

print("Starting minimal TF test...")
sys.stdout.flush()

try:
    print("Importing tensorflow...")
    sys.stdout.flush()
    import tensorflow as tf
    print("SUCCESS: tensorflow imported")
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
