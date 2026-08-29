import numpy as np
y = np.load("y_labels.npy")
print("Unique labels:", np.unique(y))
print("Min label:", np.min(y))
print("Max label:", np.max(y))
print("Shape:", y.shape)
