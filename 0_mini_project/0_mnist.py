# 머신러닝으로 mnist 돌려보기

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기

import numpy as np
from tensorflow.keras.datasets import mnist


# 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)
###


# 데이터 전처리
## reshape
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
print("after reshape x_train.shape:", x_train.shape)
print("after reshape x_test.shape:", x_test.shape)


## feature importance 관찰
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(
    x_train, y_train, train_size=0.75, test_size=0.25)







