import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
RANDOM = 44

x = np.load('./3rd_project/201214/data/all_scale_x_11025sr.npy')
y = np.load('./3rd_project/201214/data/all_scale_y_11025sr.npy')

x = x.reshape(x.shape[0], x.shape[1], 1)
y = y - 48


x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6, random_state=RANDOM)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5, random_state=RANDOM)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint # 모델 체크 포인트
from tensorflow.keras.models import load_model

model = load_model('./3rd_project/201214/model/lstm_best_weight_model.hdf5')
model.compile(loss='sparse_categorical_crossentropy', 
                        metrics=['acc'], 
                        optimizer='adam')

loss, acc = model.evaluate(x_test, y_test, batch_size=512)
print("acc:",acc)
print("loss:",loss)

