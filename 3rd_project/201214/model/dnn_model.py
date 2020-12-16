import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

x = np.load('./3rd_project/201214/data/x_val_11025sr.npy')
y = np.load('./3rd_project/201214/data/y_val_11025sr.npy')

x = x.reshape(x.shape[0], x.shape[1], 1)
y = y - 48


x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv1D(128, 3, activation='relu', padding='same', input_shape=(x_train.shape[1],1)))
model.add(Conv1D(256, 3, padding='same', activation='relu'))
model.add(Conv1D(512, 3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(37, activation='softmax')) 

model.compile(loss='sparse_categorical_crossentropy', 
                        metrics=['acc'], 
                        optimizer='adam')

ealystopping = EarlyStopping(monitor='val_loss',
                            patience=40,
                            mode='auto')

hist = model.fit(x_train, y_train, 
                    epochs=1000, 
                    batch_size=512, 
                    validation_data=(x_val, y_val), 
                    callbacks=[ealystopping])

loss, acc =model.evaluate(x_test, y_test, batch_size=512)
print("acc:",acc)
print("loss:",loss)

model.save('./3rd_project/201214/model/dnn_11025sr.h5')

# 5. 모델 학습 과정 표시하기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()


