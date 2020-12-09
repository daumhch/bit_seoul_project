import pyaudio
import numpy as np
import time
import pickle

pca_reload = pickle.load(open("./test2/model/pca.pkl",'rb'))

CHUNK = 22050
RATE = 22050

file_name = './test2/model/temp_model.pkl'
model = pickle.load(open(file_name, "rb"))

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

while(True):
    t1 = time.time()
    data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
    t2 = time.time()
    fft = np.fft.fft(data)
    magnitude = np.abs(fft)
    f = np.linspace(0, RATE, len(magnitude))
    
    pitch_index = np.where((f>110.0) & (f<1000.0))
    pitch_mag = magnitude[pitch_index]
    # print(pitch_mag.shape)
    # print('?????:',max(pitch_mag))
    y_predict = 0
    if max(pitch_mag) > 25:
        pitch_mag = pitch_mag.reshape(1,pitch_mag.shape[0])
        pitch_mag = pca_reload .transform(pitch_mag)
        y_predict = model.predict(pitch_mag)
    # print('do time: %.2fms' %((time.time()-t1)*1000), '/',y_predict )
    print("[%02d]"%y_predict, " / 1st time:%.2fms" %((t2-t1)*1000),
            '/ 2nd time:%.2fms' %((time.time()-t2)*1000))


stream.stop_stream()
stream.close()
p.terminate()



