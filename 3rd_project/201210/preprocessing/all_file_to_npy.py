import os
import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt
import pandas as pd
FIG_SIZE = (8,6)

SR = 11025
def convertFregToPitch(arr):
    return np.round(39.86*np.log10(arr/440.0) + 69.0) #수 많은 소수점 들을 하나로 합치게 해줌. Ex 130.8 130.9 130.10 을 전부 130 => 48로 단일화 즉 값들이 48로 몰링
convertFregToPitch2 = np.vectorize(convertFregToPitch)

def search(dirname):
    x_arr = []
    y_arr = []
    num = 0

    #입력받은 디렉토리의 모든 파일을 불러온다.
    filenames = os.listdir(dirname)
    for filename in filenames:

        #midi 라벨링, 파일 이름에서 추출        
        y_label = int(filename.split('-')[-2])

        #사람이 낼 수 있는 음역대로 판단한 48번 ~ 84번만 npy로 작성
        if y_label < 48 or 84 < y_label:
            continue
        full_filename = os.path.join(dirname, filename)
        full_filename = full_filename.replace('\\', '/')
        
        #리브로사를 사용하여 wav를 로드, 시작 1초간 소리가 없기 때문에 1초 후부터 시작 
        sig, sr = librosa.load(full_filename, sr=SR, offset=1.0)

        # 복소공간 값 절댓갑 취해서, magnitude 구하기
        fft = np.fft.fft(sig)
        magnitude = np.abs(fft)

        # Frequency 값 만들기
        f = np.linspace(0,sr,len(magnitude))

        # 푸리에 변환을 통과한 specturm은 대칭구조로 나와서 high frequency 부분 절반을 날려고 앞쪽 절반만 사용한다.
        left_spectrum = magnitude[:int(len(magnitude)/2)]
        left_f = f[:int(len(magnitude)/2)]

        # 불필요한 hz가 많기 때문에 48번의 hz = 130 ~ 84번 hz 1050 데이터만 사용
        # print(left_spectrum.shape) #108427
        pitch_index = np.where((left_f > 130.0) & (left_f < 1050.0)) #130 ~ 1050 헤르츠의 index 구함
        pitch_freq = left_f[pitch_index] 
        pitch_mag = left_spectrum[pitch_index] 
        # print(left_spectrum.shape) # 9000컬럼

        #주파수를 midi번호로 변경 Ex) 130.0 -> 48
        pitch_freq = convertFregToPitch2(pitch_freq)

        # 다시 48번 midi부터 84번 midi까지 슬라이싱
        start_index = np.where(pitch_freq>=48)
        pitch_freq = pitch_freq[start_index]
        pitch_mag = pitch_mag[start_index]


        #여러 미디번호들이 있지만 유니크로 보여주며 유니크 이전엔 48 48 48 48 48 48 48 이런식으로 있을 것이고 해당 인덱스로 주면 mag를 얻는다.
        freq_uniq = np.unique(pitch_freq) 

        #y 값을 평균으로 미디번호를 단일화
        tmp_arr = []
        for i in range(len(freq_uniq)):
            # print(freq_uniq[i])
            tmp_avg = np.average(pitch_mag[np.where(pitch_freq == freq_uniq[i])]) 
            tmp_arr.append(tmp_avg)

        x_arr.append(tmp_arr)
        y_arr.append(y_label)
        
        num += 1
        print(str(num)+'번째 파일')
    return np.array(x_arr), np.array(y_arr)


# x, y = search('D:/bit_seoul_project/test/data/nsynth-test/audio')
x, y = search('D:/bit_seoul_project/test/data/nsynth-train/audio')


# np.save('./3rd_project/201210/data/all_scale_x_11025sr.npy', arr=x)
# np.save('./3rd_project/201210/data/all_scale_y_11025sr.npy', arr=y)

np.save('./3rd_project/201210/data/x_train_11025sr.npy', arr=x)
np.save('./3rd_project/201210/data/y_train_11025sr.npy', arr=y)
# 28만개 train데이터의 48~84번 변환 데이터 = 146319개


