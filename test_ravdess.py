import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Utilise le chemin exact qui fonctionne sur ton PC
path = r"C:\Users\marie\Desktop\PFE Issat\data\Actor_01\03-02-01-01-01-01-01.wav"

# 2. Charger le signal audio
audio_data, sr = librosa.load(path)

# 3. Extraire les MFCCs (C'est ici qu'on veut voir le 40 !)
mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)

print("\n" + "="*30)
print(f"ANALYSE RÉUSSIE !")
print(f"Dimensions des MFCCs : {mfccs.shape}")
print("="*30)

# 4. Afficher l'onde (Optionnel : ferme la fenêtre pour voir le print)
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_data, sr=sr)
plt.title("Onde sonore - Acteur 1 (Neutre)")
plt.show()