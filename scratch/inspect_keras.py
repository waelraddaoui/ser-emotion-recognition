import tensorflow as tf
model = tf.keras.models.load_model("models/best_model.keras")
model.summary()
