import pyaudio
import numpy as np
import time

CHUNK = 1024
RATE = 44100
 
p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

while(True):
    t1 = time.time()
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    print(data.shape)
    print('do time: %.2fms' %((time.time()-t1)*1000) )

stream.stop_stream()
stream.close()
p.terminate()



