import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier
import time
import pickle

RANDOM_STATE = 44

x = np.load('./3rd_project/201214/data/all_scale_x_11025sr.npy')
y = np.load('./3rd_project/201214/data/all_scale_y_11025sr.npy')


#train, test, val 을 구분하기 위한 train_test_split
x_train, x_test ,y_train , y_test= train_test_split(
    x, y, train_size = 0.6, random_state=RANDOM_STATE)
x_test, x_val ,y_test , y_val= train_test_split(
    x_test, y_test, train_size = 0.5, random_state=RANDOM_STATE)

print('x_train.shape:',x_train.shape)
print('x_val.shape:',x_val.shape)
print('x_test.shape:',x_test.shape)


model = pickle.load(open('./3rd_project/201214/model/lgbm_best_weight_model.hdf5','rb'))

score = model.score(x_test, y_test)
print("score:", score)







