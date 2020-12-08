import time
import sounddevice as sd
import numpy as np
import copy
import socket

duration = 3  # seconds

while True:
    
    present_wave = []
    compare_wave = []

    def print_sound(indata, outdata, frames, time, status):
        volume_norm = np.linalg.norm(indata)*10
        print("|" * int(volume_norm))

    with sd.Stream(callback=print_sound):
        sd.sleep(duration * 1000)


    time.sleep(0.01)

# sounddevice를 설치하고
# 실시간 마이크 입력이 정상 동작 하는 것을 확인함