import pyaudio
import numpy as np
import time
import pickle
import queue
import datetime

CHUNK = 44100
RATE = 44100


class MicStream(object):
    # 클래스 선언 할 때
    def __init__(self, rate, chunk):
        print('class init')
        self.rate = rate
        self.chunk = chunk
        self.buff = queue.Queue()
        self.closed = True
        self.t1 = 0

    # 클래스 시작 할 때
    def __enter__(self):
        print('class enter')
        self.closed = False
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format = pyaudio.paFloat32,
            channels=1,
            rate = self.rate,
            input = True,
            frames_per_buffer = self.chunk,
            stream_callback = self.fill_buffer)
        return self

    # 클래스 종료 할 때
    def __exit__(self, type, value, traceback):
        print('class exit')
        self.closed = True
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.buff.put(None)
        self.audio_interface.terminate()

    def fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self.t1 = time.time()
        self.buff.put(in_data)
        return None, pyaudio.paContinue

    # 사용자 함수
    def record(self):
        while not self.closed:
            chunk = self.buff.get()
            if chunk is None:
                return
            data = [chunk]
            # print('data:',data)
            # print('class recored')
            print('do time:', datetime.datetime.now())

def main():
    with MicStream(RATE, CHUNK) as stream:
        stream.record()

if __name__ == '__main__':
    main()


# sr = 44100 = 1초, chunk =44100이면 1초 걸릴 수 밖에 없다
# 그럼에도 sr을 낮춰서 해보자
