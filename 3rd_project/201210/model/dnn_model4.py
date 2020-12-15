
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
# from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU, PReLU

import time
import pickle


RANDOM_STATE = 44

x = np.load('./3rd_project/201210/data/all_scale_x_11025sr.npy')
y = np.load('./3rd_project/201210/data/all_scale_y_11025sr.npy')
y = y - 48

x = x.reshape(x.shape[0], x.shape[1], 1)

#train, test, val 을 구분하기 위한 train_test_split
x_train, x_test ,y_train , y_test= train_test_split(
    x, y, train_size = 0.6, random_state=RANDOM_STATE)
x_test, x_val ,y_test , y_val= train_test_split(
    x_test, y_test, train_size = 0.5, random_state=RANDOM_STATE)

print('x_train.shape:',x_train.shape)
print('x_val.shape:',x_val.shape)
print('x_test.shape:',x_test.shape)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dense, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, MaxPooling1D

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU, ELU, LeakyReLU, PReLU
from tensorflow.keras.activations import relu, selu, elu, softmax, sigmoid

from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam




model = Sequential()
model.add(Conv1D(64, 3, activation='selu', padding='same', input_shape=(x_train.shape[1],1)))
model.add(Conv1D(256, 3, padding='same', activation='selu'))
model.add(Conv1D(512, 3, padding='same', activation='selu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(37, activation='softmax')) #ouput 

# 함수형 모델 마무리
model.compile(optimizer='adam',
                metrics=['acc'],
                loss='sparse_categorical_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss',
                                patience=10,
                                mode='auto')

model.fit(x_train, y_train, 
            epochs=1000, 
            batch_size=512, 
            validation_data=(x_val, y_val), 
            callbacks=[early_stopping])

loss, acc=model.evaluate(x_test, y_test, batch_size=512)
print("acc",acc)
print("loss",loss)



