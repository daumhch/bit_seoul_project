import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import datetime
CHUNK = 11025 
RATE = 44100 
SR = 44100
OFFSET = 48
MIDI_NUMS = [i for i in range(48,85)]
MODEL = pickle.load(open('./model/modelLoad/modelFolder/lgbm_11025sr.dat', 'rb'))
SCALE = ['C','D','E', 'F', 'G', 'A', 'B']

def convertFregToPitch(arr):
    return np.round(39.86*np.log10(arr/440.0) + 69.0) #수 많은 소수점 들을 하나로 합치게 해줌. Ex 130.8 130.9 130.10 을 전부 130 => 48로 단일화 즉 값들이 48로 몰링
convertFregToPitch2 = np.vectorize(convertFregToPitch)

def transe2(y):
    tmp = (int(y) - OFFSET) % 12
    result = [0 for i in range(7)]
    if tmp == 0:
        result[0] +=2
    elif tmp == 2:
        result[1] +=2
    elif tmp == 4:
        result[2] +=2
    elif tmp == 5:
        result[3] +=2
    elif tmp == 6:
        result[4] +=2
    elif tmp == 8:
        result[5] +=2
    elif tmp == 10:
        result[6] +=2
    return result

def transe(y):
    tmp = (int(y) - OFFSET) % 12
    result = ''
    if tmp == 0:
        result = '도'
    elif tmp == 2:
        result = '레'
    elif tmp == 4:
        result = '미'
    elif tmp == 5:
        result = '파'
    elif tmp == 6:
        result = '솔'
    elif tmp == 8:
        result = '라'
    elif tmp == 10:
        result = '시'
    return result
    
def preprocessing(data):
    fft = np.fft.fft(data)
    magnitude = np.abs(fft)
    f = np.linspace(0,SR,len(magnitude)) 
    left_spectrum = magnitude[:int(len(magnitude)/2)]
    left_f = f[:int(len(magnitude)/2)]
    pitch_index = np.where((left_f > 130.0) & (left_f < 1050.0)) #130 ~ 1050 헤르츠의 index 구함
    pitch_freq = left_f[pitch_index] #x축 
    pitch_mag = left_spectrum[pitch_index] #y축

    pitch_freq = convertFregToPitch2(pitch_freq)

    start_index = np.where(pitch_freq>=48)
    pitch_freq = pitch_freq[start_index]
    pitch_mag = pitch_mag[start_index]
    freq_uniq = np.unique(pitch_freq)


    tmp_arr = []
    for i in range(len(freq_uniq)):
        tmp_avg = np.average(pitch_mag[np.where(pitch_freq == freq_uniq[i])]) # 48을 가진 index 들을 모두 가져와서 avg
        tmp_arr.append(tmp_avg)

    #백색소음일 때의 파워 값은 대략 8 내외 8보다 작으면 소리 입력이 없는 것으로 판단.
    #소리가 있다면 predict
    if max(pitch_mag) < 8:
        result = np.array(tmp_arr)
        return result, False
    else:
        result = np.array(tmp_arr)
        result = result.reshape(1,result.shape[0])
        return result, True


 
    # plt.figure(figsize=(16,10))
    # plt.xlim([0,6])
    # plt.ylim([-1,2])
    # # plt.plot(left_f,left_spectrum)
    # plt.plot(SCALE,result)
    # plt.savefig('./03.png', dpi=50)
    # print("한 번 출력까지 걸린 시간 : ",datetime.datetime.now() - start_time)

if __name__ == '__main__':
    
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paFloat32,channels=1,rate=RATE,input=True,
        frames_per_buffer=CHUNK)
    
    while(True):
        start_time = datetime.datetime.now()
        data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
        print("stream 걸린 시간 : ",datetime.datetime.now() - start_time)
        pre_data, is_sound  = preprocessing(data)
        result = [0 for i in range(7)]
        if is_sound:
            y_predict = MODEL.predict(pre_data)
            result = transe2(y_predict)
            print()
        else:
            print('소리 없음')
    stream.stop_stream()
    stream.close()
    p.terminate()
