import numpy as np
import matplotlib.pyplot as plt
import librosa

file_path = './test/data/nsynth-test/audio/vocal_acoustic_000-060-050.wav'

y, sr = librosa.load(file_path, sr=8000, offset=1.0, duration=2.0)
print("sr:",sr)
print("y.shape",y.shape)
fft = np.fft.fft(y)
magnitude = np.abs(fft)
f = np.linspace(0, sr, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

plt.plot(left_f, left_spectrum)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()





# print(type(left_f))
# print(type(left_spectrum))
# print(left_f.shape)
# print(left_spectrum.shape)
pitch_index = np.where((left_f>250.0) & (left_f<500.0))
pitch_freq = left_f[pitch_index]
pitch_mag = left_spectrum[pitch_index]
# print(pitch_freq)
# print(pitch_mag)

mag_max = np.max(pitch_mag)
max_index = np.where(pitch_mag == mag_max)
print(max_index)
print(pitch_freq[max_index])
def convertFregToPitch(arr):
    return 39.86*np.log10(arr/440.0) + 69.0
print(convertFregToPitch(pitch_freq[max_index]))

# 도(C4)~시(B4) 까지의 주파수 범위로 좁혀서, 최대값의 주파수를 확인
