import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import timeit

# 사용자 함수 - 주파수를 미디번호로 바꾸기 A4=440Hz=69번
def convertFregToPitch(arr):
    return np.round(39.86*np.log10(arr/440.0) + 69.0)
convertFregToPitch2 = np.vectorize(convertFregToPitch)

path_dir = './test/data/nsynth-test/audio'
file_list = os.listdir(path_dir)
print("len(file_list):",len(file_list))

# cnt = 43
test_wav_data = []
test_wav_target = []
test_wav_filename = []

for cnt in range(len(file_list)):
    # 파일 읽어서 FFT
    y, sr = librosa.load(path_dir+'/'+file_list[cnt])
    fft = np.fft.fft(y)/len(y)
    magnitude = np.abs(fft)
    f = np.linspace(0, sr, len(magnitude))
    left_spectrum = magnitude[:int(len(magnitude)/2)]
    left_f = f[:int(len(magnitude)/2)]

    # 미디번호 21~108번에 매칭되는 주파수(27Hz~4186Hz)보다 약간 넓게 슬라이싱
    pitch_index = np.where((left_f>0.0) & (left_f<4200.0))

    pitch_freq = left_f[pitch_index]
    pitch_freq = convertFregToPitch2(pitch_freq)
    pitch_mag = left_spectrum[pitch_index]

    start_index = np.where(pitch_freq>=21)
    pitch_freq = pitch_freq[start_index]
    pitch_mag = pitch_mag[start_index]

    freq_uniq = np.unique(pitch_freq)

    temp_arr = []
    for i in range(freq_uniq.shape[0]):
        temp_avg = np.average(pitch_mag[np.where(pitch_freq==freq_uniq[i])])
        temp_arr = np.insert(temp_arr, i, temp_avg)

    #print(temp_arr.shape)
    test_wav_data.append(temp_arr)

    pitch = int(file_list[cnt][-11:-8])
    test_wav_target = np.insert(test_wav_target, cnt, pitch)
    test_wav_filename.append(file_list[cnt])
    print(pitch)

test_wav_data = np.array(test_wav_data)
# print(test_wav_data[:3])
print(test_wav_data.shape)
print(test_wav_target.shape)


np.save('./test/data/test_wav_data.npy', arr=test_wav_data)
np.save('./test/data/test_wav_target.npy', arr=test_wav_target)
np.save('./test/data/test_wav_filename.npy', arr=test_wav_filename)

# 주파수 분석 결과를 미디번호로 바꾼 데이터를 npy로 저장함

