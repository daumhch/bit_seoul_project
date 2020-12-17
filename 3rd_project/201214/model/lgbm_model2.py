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

start_time = time.time()

def model(
    num_leaves,
    max_depth,
    learning_rate,
    n_estimators):
    model = LGBMClassifier(n_jobs=6,
            num_leaves = int(num_leaves),
            max_depth = int(max_depth),
            learning_rate = learning_rate,
            n_estimators = int(n_estimators)
            )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("score", acc)
    return acc


from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'num_leaves':np.array(range(256,1024,16)), # (16, 1024),
    'max_depth':np.array(range(7,20,1)), # (7, 20),
    'learning_rate':np.arange(0.001,0.1,0.01), # (0.001, 0.1),
    'n_estimators':np.array(range(50,200,10)), # (50, 200),
}
start_time = time.time()
RSCV = RandomizedSearchCV(LGBMClassifier(metrics=['multi_logloss']), 
                        parameters, cv=5,
                        random_state=RANDOM_STATE,
                        verbose=2)
RSCV.fit(x_train,y_train, eval_set=[(x_val, y_val)])
score = RSCV.score(x_test, y_test)
print("score:", score)
best_params = RSCV.best_params_
print("최적의 파라미터:", best_params)
print("RSCV fit 소요 시간: %.3fs" %((time.time() - start_time)) )



best_model = RSCV.best_estimator_
score = best_model.score(x_test, y_test)
print("best_model score:", score)

pickle.dump(best_model, open('./3rd_project/201214/model/lgbm_best_weight_model.hdf5', 'wb'))
