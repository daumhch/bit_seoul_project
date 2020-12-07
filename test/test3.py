import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import timeit

# file_path = './data/same_pitch_diff_velo/vocal_acoustic_000-060-050.wav'
# file_path = './data/same_pitch_diff_velo/vocal_acoustic_000-060-075.wav'

# file_path = './data/nsynth-test/audio/vocal_acoustic_000-060-050.wav'
file_path = './test/data/nsynth-test/audio/organ_electronic_007-009-050.wav'


start_time = timeit.default_timer() # 시작 시간 체크

y, sr = librosa.load(file_path)

fft = np.fft.fft(y)/len(y)
magnitude = np.abs(fft)
f = np.linspace(0, sr, len(magnitude))
left_spectrum = magnitude[:int(len(magnitude)/2)]
left_f = f[:int(len(magnitude)/2)]

pitch_index = np.where((left_f>20.0) & (left_f<4200.0))
print(pitch_index)

pitch_freq = left_f[pitch_index]
pitch_mag = left_spectrum[pitch_index]
print(pitch_freq)
print(pitch_mag)

def convertFregToPitch(arr):
    return np.round(39.86*np.log10(arr/440.0) + 69.0)
convertFregToPitch2 = np.vectorize(convertFregToPitch)
pitch_freq = convertFregToPitch2(pitch_freq)

start_index = np.where(pitch_freq>=21)
print("start_index:",start_index)
pitch_freq = pitch_freq[start_index]
pitch_mag = pitch_mag[start_index]
print(pitch_freq)
print(pitch_mag)
print("pitch_freq.shape:",pitch_freq.shape)
print("pitch_mag.shape:",pitch_mag.shape)

freq_uniq = np.unique(pitch_freq)
print(freq_uniq[0])

temp_arr = []
for cnt in range(freq_uniq.shape[0]):
    temp_avg = np.average(pitch_mag[np.where(pitch_freq==freq_uniq[cnt])])
    temp_arr = np.insert(temp_arr, cnt, temp_avg)

print(temp_arr.shape)
plt.plot(freq_uniq, temp_arr)
plt.xlabel("MIDI number")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()
terminate_time = timeit.default_timer() # 종료 시간 체크  
print("%f초 걸렸습니다." % (terminate_time - start_time)) 

# 주파수별로 중복된 값을 합쳐서
# 미디번호 21~108 사이 그래프가 되도록 정리함

# file_path = './data/same_pitch_diff_inst/reed_acoustic_037-060-127.wav'

# print(int(file_path[-11:-8]))
# 60


np.save('./test/data/mangi_predict_data.npy', arr=temp_arr)

