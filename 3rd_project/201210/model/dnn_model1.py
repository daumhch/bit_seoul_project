
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




def build_model(node_value1=256,
                kernel_num1=3,
                node_value2=128,
                kernel_num2=3,
                node_value3=512,
                kernel_num3=3,
                node_value4=150
                ):
    model = Sequential()
    model.add(Conv1D(node_value1, kernel_num1, activation='relu', padding='same', 
                input_shape=(x_train.shape[1],x_train.shape[2])))
    model.add(Conv1D(node_value2, kernel_num2, padding='same', activation='relu'))
    model.add(Conv1D(node_value3, kernel_num3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(node_value4, activation='relu'))
    model.add(Dense(37, activation='softmax')) #ouput 
    
    # 함수형 모델 마무리
    model.compile(optimizer='adam',
                    metrics=['acc'],
                    loss='sparse_categorical_crossentropy')

    # 모델 정보 출력
    print(model.summary() )
    return model

def create_hyperparameters():
    epochs = [100, 200]

    node_value1=[128, 256, 512]
    kernel_num1=[3, 6, 9]
    node_value2=[128, 256, 512]
    kernel_num2=[3, 6, 9]
    node_value3=[128, 256, 512]
    kernel_num3=[3, 6, 9]
    node_value4=[128, 256, 512]

    return_parameter = {"epochs":epochs,
                    'node_value1':node_value1,
                    'kernel_num1':kernel_num1,
                    'node_value2':node_value2,
                    'kernel_num2':kernel_num2,
                    'node_value3':node_value3,
                    'kernel_num3':kernel_num3,
                    'node_value4':node_value4
                    }
    return return_parameter


from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

wrapper_model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(wrapper_model, hyperparameters, cv=5)


from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(monitor='val_loss',
                                patience=10,
                                mode='auto')
search.fit(x_train, y_train, 
            callbacks=[early_stopping])

print("search.best_params_:\n",search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어:", acc)


