import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# file_path = './data/nsynth-test/audio/vocal_acoustic_000-060-050.wav'
file_path = './test/data/048.wav'

y, sr = librosa.load(file_path)

fft = np.fft.fft(y)/len(y)
magnitude = np.abs(fft)
f = np.linspace(0, sr, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]
print(type(left_f))
print(left_f.shape)
# left_f = 39.86*math.log10(left_f/440.0) + 69
mag_max = np.max(left_spectrum)
print(mag_max)
max_index = np.where(left_spectrum==mag_max)

print(left_f[max_index])

def convertPitch(arr):
    if arr != 0:
        return 39.86*np.log10(arr/440.0) + 69.0
convertPitch2 = np.vectorize(convertPitch)
left_f = convertPitch2(left_f)

print(left_f[max_index])

plt.scatter(left_f, left_spectrum)
plt.xlabel("MIDI number")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()


# 주파수를 미디번호로 바꾸는 공식 찾음
# 또한 array에 커스텀함수를 돌리려면 vectorize하면 됨