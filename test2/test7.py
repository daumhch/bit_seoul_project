import pyaudio
import numpy as np
import time
import pickle

CHUNK = 11025
RATE = 11025

file_name = './test2/model/temp_model.pkl'
model = pickle.load(open(file_name, "rb"))

p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

while(True):
    t1 = time.time()
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    data = data.reshape(1,data.shape[0])
    y_predict = model.predict(data)
    print('do time: %.2fms' %((time.time()-t1)*1000), '/',y_predict )

stream.stop_stream()
stream.close()
p.terminate()



