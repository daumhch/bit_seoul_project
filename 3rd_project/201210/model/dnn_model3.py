
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import MaxPooling1D, Dropout


# model = Sequential()
# model.add(Conv1D(64, 9, input_shape=(x_train.shape[1],x_train.shape[2])) )
# model.add(Conv1D(64, 9, padding='same', strides=2, activation='relu') )
# model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
# model.add(Dropout(0.2))

# model.add(Conv1D(128, 9, padding='same', strides=1, activation='relu') )
# model.add(Conv1D(128, 9, padding='same', strides=1, activation='relu') )
# model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
# model.add(Dropout(0.3))

# model.add(Conv1D(256, 9, padding='same', strides=1, activation='relu') )
# model.add(Conv1D(256, 9, padding='same', strides=1, activation='relu') )
# model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
# model.add(Dropout(0.4))

# model.add(Flatten())
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(37, activation='softmax', name='outputs'))


def build_model(
    node_1st=64,
    kernel_1st=9,
    dropout_1st=0.2,
    node_2nd=128,
    kernel_2nd=9,
    dropout_2nd=0.3,
    node_3rd=256,
    kernel_3rd=9,
    dropout_3rd=0.4,
    dense_node=64,
    dense_layers=1
    ):
    node_1st=int(node_1st)
    kernel_1st=int(kernel_1st)
    dropout_1st=int(dropout_1st)
    node_2nd=int(node_2nd)
    kernel_2nd=int(kernel_2nd)
    dropout_2nd=int(dropout_2nd)
    node_3rd=int(node_3rd)
    kernel_3rd=int(kernel_3rd)
    dropout_3rd=int(dropout_3rd)
    dense_node=int(dense_node)
    dense_layers=int(dense_layers)

    model = Sequential()
    model.add(Conv1D(node_1st, kernel_1st, input_shape=(x_train.shape[1],x_train.shape[2])) )
    model.add(Conv1D(node_1st, kernel_1st, padding='same', strides=2, activation='relu') )
    model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
    model.add(Dropout(dropout_1st))

    model.add(Conv1D(node_2nd, kernel_2nd, padding='same', strides=1, activation='relu') )
    model.add(Conv1D(node_2nd, kernel_2nd, padding='same', strides=1, activation='relu') )
    model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
    model.add(Dropout(dropout_2nd))

    model.add(Conv1D(node_3rd, kernel_3rd, padding='same', strides=1, activation='relu') )
    model.add(Conv1D(node_3rd, kernel_3rd, padding='same', strides=1, activation='relu') )
    model.add(MaxPooling1D(pool_size=2, padding='valid', strides=2))
    model.add(Dropout(dropout_3rd))

    model.add(Flatten())
    for cnt in range(dense_layers):
        model.add(Dense(dense_node, activation = 'relu'))
    model.add(Dense(37, activation='softmax', name='outputs'))
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )
    ealystopping = EarlyStopping(monitor='val_loss',
                        patience=40,
                        mode='auto')
    model.fit(x_train, y_train, 
        epochs=1000, 
        batch_size=512, 
        validation_data=(x_val, y_val), 
        callbacks=[ealystopping])
    # return model
    loss, acc=model.evaluate(x_test, y_test, batch_size=512)
    return acc


# model = build_model()
# model.summary()

from bayes_opt import BayesianOptimization
parameters = {
    'node_1st':(64, 256),
    'kernel_1st':(6, 12),
    'dropout_1st':(0.2, 0.5),
    'node_2nd':(64, 256),
    'kernel_2nd':(6, 12),
    'dropout_2nd':(0.2, 0.5),
    'node_3rd':(64, 256),
    'kernel_3rd':(6, 12),
    'dropout_3rd':(0.2, 0.5),
    'dense_node':(64, 256),
    'dense_layers':(1,6)
}
bo = BayesianOptimization(f=build_model, pbounds=parameters, verbose=1, random_state=RANDOM_STATE)
bo.maximize(init_points=4, n_iter=100, acq='ei')
print('===============')
print(bo.max)
print('===============')

'''
def create_hyperparameters():
    epochs = [256, 512, 1024]
    node_1st=[64, 128, 256]
    kernel_1st=[3, 6, 9, 12]
    dropout_1st=[0.2, 0.3, 0.4]
    node_2nd=[64, 128, 256]
    kernel_2nd=[3, 6, 9, 12]
    dropout_2nd=[0.2, 0.3, 0.4]
    node_3rd=[64, 128, 256]
    kernel_3rd=[3, 6, 9, 12]
    dropout_3rd=[0.2, 0.3, 0.4]
    dense_node=[64,128,256,512]
    dense_layers=[1,2,3,4,5,6]

    return_parameter = {
                    "epochs":epochs,
                    'node_1st':node_1st,
                    'kernel_1st':kernel_1st,
                    'dropout_1st':dropout_1st,
                    'node_2nd':node_2nd,
                    'kernel_2nd':kernel_2nd,
                    'dropout_2nd':dropout_2nd,
                    'node_3rd':node_3rd,
                    'kernel_3rd':kernel_3rd,
                    'dropout_3rd':dropout_3rd,
                    'dense_node':dense_node,
                    'dense_layers':dense_layers
                    }
    return return_parameter

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

wrapper_model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()
search = RandomizedSearchCV(wrapper_model, hyperparameters, cv=3)

from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
early_stopping = EarlyStopping(monitor='loss',
                                patience=10,
                                mode='auto')
search.fit(x_train, y_train, 
            callbacks=[early_stopping])

print("search.best_params_:\n",search.best_params_)
acc = search.score(x_test, y_test)
print("최종 스코어:", acc)
'''

# from tensorflow.keras.callbacks import EarlyStopping # 조기 종료
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=50,
#     mode='auto',
#     verbose=2)

# model.fit(
#     x_train, y_train,
#     epochs=1000,
#     batch_size=512,
#     verbose=1,
#     validation_split=0.2,
#     callbacks=[early_stopping]
#     )

# 4. 평가, 예측
# loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)
# print("loss: ", loss)
# print("accuracy: ", accuracy)
