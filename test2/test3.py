import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time

SR = 11025

file_path = './test/data/nsynth-test/audio/vocal_acoustic_000-060-050.wav'
y, sr = librosa.load(file_path, 
                        sr=SR, 
                        offset=1.0, 
                        duration=1.0)
print("sr:",sr)
print("y.shape",y.shape)
plt.figure(figsize=(10, 5))
librosa.display.waveplot(y, sr=sr)
plt.show()


fft = np.fft.fft(y)
magnitude = np.abs(fft)
print("magnitude.shape:", magnitude.shape)
f = np.linspace(0, sr, len(magnitude))
plt.plot(f, magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()


# np.save('./test2/data/test_predict_data.npy', arr=pitch_mag)


# pitch_index = np.where((f>10.0) & (f<4200.0))
# pitch_freq = f[pitch_index].astype(np.int16)
# pitch_mag = magnitude[pitch_index]
# plt.plot(pitch_freq, pitch_mag)