import pyaudio
import time
import pylab
import numpy as np
import pickle

file_name = './test2/model/temp_model.pkl'

class SWHear(object):
    """
    The SWHear class is made to provide access to continuously recorded
    (and mathematically processed) microphone data.
    """
    model = pickle.load(open(file_name, "rb"))

    def __init__(self,device=None,startStreaming=True):
        """fire up the SWHear class."""
        print(" -- initializing SWHear")

        self.chunk = 44100 # number of data points to read at a time
        self.rate = 44100 # time resolution of the recording device (Hz)

        # for tape recording (continuous "tape" of recent audio)
        self.tapeLength=2 #seconds
        self.tape=np.empty(self.rate*self.tapeLength)*np.nan

        self.p=pyaudio.PyAudio() # start the PyAudio class
        if startStreaming:
            self.stream_start()

    ### LOWEST LEVEL AUDIO ACCESS
    # pure access to microphone and stream operations
    # keep math, plotting, FFT, etc out of here.

    def convert_fft(self, data):
        fft = np.fft.fft(data)
        magnitude = np.abs(fft)
        f = np.linspace(0, self.rate, len(magnitude))
        pitch_index = np.where((f>10.0) & (f<4200.0))
        # pitch_freq = f[pitch_index].astype(np.int16)
        pitch_mag = magnitude[pitch_index]
        y_predict = 0
        if max(pitch_mag) > 50:
            pitch_mag = pitch_mag.reshape(1,pitch_mag.shape[0])
            y_predict = self.model.predict(pitch_mag)
        print(y_predict)
        return y_predict

    def stream_read(self):
        """return values for a single chunk"""
        # data = np.fromstring(self.stream.read(self.chunk),dtype=np.int16)
        data = np.frombuffer(self.stream.read(self.chunk),dtype=np.float32)
        data = self.convert_fft(data)
        return data


    def stream_start(self):
        """connect to the audio device and start a stream"""
        print(" -- stream started")
        self.stream=self.p.open(format=pyaudio.paFloat32,
                                channels=1,
                                rate=self.rate,
                                input=True,
                                frames_per_buffer=self.chunk)

    def stream_stop(self):
        """close the stream but keep the PyAudio instance alive."""
        if 'stream' in locals():
            self.stream.stop_stream()
            self.stream.close()
        print(" -- stream CLOSED")

    def close(self):
        """gently detach from things."""
        self.stream_stop()
        self.p.terminate()

    ### TAPE METHODS
    # tape is like a circular magnetic ribbon of tape that's continously
    # recorded and recorded over in a loop. self.tape contains this data.
    # the newest data is always at the end. Don't modify data on the type,
    # but rather do math on it (like FFT) as you read from it.

    def tape_add(self):
        """add a single chunk to the tape."""
        self.tape[:-self.chunk]=self.tape[self.chunk:]
        self.tape[-self.chunk:]=self.stream_read()

    def tape_flush(self):
        """completely fill tape with new data."""
        readsInTape=int(self.rate*self.tapeLength/self.chunk)
        print(" -- flushing %d s tape with %dx%.2f ms reads"%\
                  (self.tapeLength,readsInTape,self.chunk/self.rate))
        for i in range(readsInTape):
            self.tape_add()

    def tape_forever(self,plotSec=0.25):
        t1=0
        try:
            while True:
                self.tape_add()
                if (time.time()-t1)>plotSec:
                    t1=time.time()
                    self.tape_plot()
        except:
            print(" ~~ exception (keyboard?)")
            return

    def tape_plot(self,saveAs="03.png"):
        """plot what's in the tape."""
        pylab.plot(np.arange(len(self.tape))/self.rate,self.tape)
        # pylab.axis([0,self.tapeLength,-2**16/2,2**16/2])
        pylab.axis([0,self.tapeLength, 0,200])
        if saveAs:
            t1=time.time()
            pylab.savefig(saveAs,dpi=50)
            print("plotting saving took %.02f ms"%((time.time()-t1)*1000))
        else:
            pylab.show()
            print() #good for IPython
        pylab.close('all')

if __name__=="__main__":
    ear=SWHear()
    ear.tape_forever()
    ear.close()
    print("DONE")