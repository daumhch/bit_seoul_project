from flask import Flask, Response, render_template
import pyaudio
import numpy as np
import pickle

app = Flask(__name__)


file_name = './test2/model/temp_model.pkl'
model = pickle.load(open(file_name, "rb"))
pca_reload = pickle.load(open("./test2/model/pca.pkl",'rb'))

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 44100
audio1 = pyaudio.PyAudio()

@app.route('/audio')
def audio():
    # start Recording
    def sound():
        stream=audio1.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("recording...")
        #frames = []
        while True:
            # data = stream.read(CHUNK)
            data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
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
            yield(str(y_predict))
    return Response(sound())

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(threaded=True)
