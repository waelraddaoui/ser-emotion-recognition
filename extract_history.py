import pickle

with open('models/training_history_gpu.pkl', 'rb') as f:
    data = pickle.load(f)

epochs = len(data['accuracy'])
print("Epoch,Training Accuracy (%),Validation Accuracy (%),Training Loss,Validation Loss")

# Print up to 15 epochs evenly spaced to fit nicely in a table
step = max(1, epochs // 15)

for i in range(0, epochs, step):
    acc = data['accuracy'][i] * 100
    val_acc = data['val_accuracy'][i] * 100
    loss = data['loss'][i]
    val_loss = data['val_loss'][i]
    print(f"Epoch {i+1},{acc:.2f}%,{val_acc:.2f}%,{loss:.4f},{val_loss:.4f}")

# Make sure the very last epoch is always printed
if (epochs - 1) % step != 0:
    i = epochs - 1
    acc = data['accuracy'][i] * 100
    val_acc = data['val_accuracy'][i] * 100
    loss = data['loss'][i]
    val_loss = data['val_loss'][i]
    print(f"Epoch {epochs},{acc:.2f}%,{val_acc:.2f}%,{loss:.4f},{val_loss:.4f}")
