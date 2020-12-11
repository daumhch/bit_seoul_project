import pygame
import sys
import pyaudio
import pickle
import numpy as np
import live_to_scale
# step1 : set screen, fps
# step2 : show dino, jump dino
# step3 : show tree, move tree
import datetime

pygame.init()
pygame.display.set_caption('Jumping dino')
MAX_WIDTH = 800
MAX_HEIGHT = 400

CHUNK = 5000
RATE = 44100 
SR = 44100
OFFSET = 48
MIDI_NUMS = [i for i in range(48,85)]
MODEL = pickle.load(open('./model/modelLoad/modelFolder/lgbm_11025sr.dat', 'rb'))
SCALE = ['C','D','E', 'F', 'G', 'A', 'B']

#오디오 스트림 생성
p=pyaudio.PyAudio()
stream=p.open(format=pyaudio.paFloat32,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK)
 
def main():
    # set screen, fps
    screen = pygame.display.set_mode((MAX_WIDTH, MAX_HEIGHT))
    fps = pygame.time.Clock()
 
    # dino
    imgDino1 = pygame.image.load('images/dino1.png')
    imgDino2 = pygame.image.load('images/dino2.png')
    dino_height = imgDino1.get_size()[1]
    dino_bottom = MAX_HEIGHT - dino_height
    dino_x = 50
    dino_y = dino_bottom
    jump_top = 200
    leg_swap = True
    is_bottom = True
    is_go_up = False
 
    # tree
    imgTree = pygame.image.load('images/tree.png')
    tree_height = imgTree.get_size()[1]
    tree_x = MAX_WIDTH
    tree_y = MAX_HEIGHT - tree_height
 
    while True:
        screen.fill((255, 255, 255))
        start_time = datetime.datetime.now()

        #오디오 스트림을 청크단위로 읽기
        data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
        
        print("stream 걸린 시간 : ",datetime.datetime.now() - start_time)
        
        #predidct를 위한 전처리
        x_predict, is_sound  = live_to_scale.preprocessing(data)
 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # elif event.type != 0:
            #     if is_bottom:
            #         is_go_up = True
            #         is_bottom = False

        # 백색소음이 아니라면 점프!
        if is_sound:
            y_predict = MODEL.predict(x_predict)
            result = live_to_scale.transe(y_predict)
            print(result)
            if result == '미':
                is_go_up = True
                is_bottom = False
        # dino move
        if is_go_up:
            dino_y -= 10.0
        elif not is_go_up and not is_bottom:
            dino_y += 10.0
 
        # dino top and bottom check
        if is_go_up and dino_y <= jump_top:
            is_go_up = False
 
        if not is_bottom and dino_y >= dino_bottom:
            is_bottom = True
            dino_y = dino_bottom
 
        # tree move
        tree_x -= 12.0
        if tree_x <= 0:
            tree_x = MAX_WIDTH
 
        # draw tree
        screen.blit(imgTree, (tree_x, tree_y))
 
        # draw dino
        if leg_swap:
            screen.blit(imgDino1, (dino_x, dino_y))
            leg_swap = False
        else:
            screen.blit(imgDino2, (dino_x, dino_y))
            leg_swap = True
 
        # update
        pygame.display.update()
        fps.tick(10000)
 
 
if __name__ == '__main__':
    main()

