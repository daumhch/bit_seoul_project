# https://kcal2845.tistory.com/35

# https://people.csail.mit.edu/hubert/pyaudio/docs/

import pyaudio
import numpy as np
import pickle

file_name = './test/model/temp_model.pkl'
model = pickle.load(open(file_name, "rb"))

def convertFregToPitch(arr):
    return np.round(39.86*np.log10(arr/440.0) + 69.0)
convertFregToPitch2 = np.vectorize(convertFregToPitch)


CHUNK = 2**10 # 불러오는 음성데이터 갯수
RATE = 8000 # 샘플링 레이트
sr = RATE

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16, # 비트 깊이
                channels=1,
                rate=RATE, # 샘플링 레이트
                input=True, # 입력 스트림인지
                frames_per_buffer=CHUNK)

temp_arr = []

while(True):
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    print(data.shape)

stream.stop_stream()
stream.close()
p.terminate()


# 느리고,
# 정확하지 않다
# 방법을 찾아보자