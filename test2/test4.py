import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import timeit

# 사용자 함수 - 주파수를 미디번호로 바꾸기 A4=440Hz=69번

path_dir = './test/data/nsynth-test/audio'
file_list = os.listdir(path_dir)
print("len(file_list):",len(file_list))

# cnt = 43
test_wav_data = []
test_wav_target = []
test_wav_filename = []

for cnt in range(len(file_list)):
    y, sr = librosa.load(path_dir+'/'+file_list[cnt], sr=44100, offset=1.0, duration=1.0)
    fft = np.fft.fft(y)/len(y)
    magnitude = np.abs(fft)
    f = np.linspace(0, sr, len(magnitude))
    pitch_index = np.where((f>10.0) & (f<4200.0))
    pitch_freq = f[pitch_index].astype(np.int16)
    pitch_mag = magnitude[pitch_index]
    test_wav_data.append(pitch_mag)

    pitch = int(file_list[cnt][-11:-8])
    test_wav_target = np.insert(test_wav_target, cnt, pitch)
    test_wav_filename.append(file_list[cnt])
    print(cnt,'/',pitch)

test_wav_data = np.array(test_wav_data)
print(test_wav_data.shape)
print(test_wav_target.shape)

np.save('./test2/data/test_wav_data.npy', arr=test_wav_data)
np.save('./test2/data/test_wav_target.npy', arr=test_wav_target)
np.save('./test2/data/test_wav_filename.npy', arr=test_wav_filename)

# 주파수 분석 결과를 미디번호로 바꾼 데이터를 npy로 저장함
