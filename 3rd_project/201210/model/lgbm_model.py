import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier
import time
import pickle

RANDOM_STATE = 44

x = np.load('./3rd_project/201210/data/all_scale_x_11025sr.npy')
y = np.load('./3rd_project/201210/data/all_scale_y_11025sr.npy')


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


'''
#범위
from bayes_opt import BayesianOptimization

parameters = {
    'num_leaves':(16, 1024),
    'max_depth':(7, 20),
    'learning_rate':(0.001, 0.1),
    'n_estimators':(50, 200),
}

bo = BayesianOptimization(f=model, pbounds=parameters, verbose=1, random_state=RANDOM_STATE)

# 메소드를 이용해 최대화 과정 수행
# init_points :  초기 Random Search 개수
# n_iter : 반복 횟수 (몇개의 입력값-함숫값 점들을 확인할지! 많을수록 정확한 값을 얻을 수 있다.)
# acq : Acquisition Function들 중 Expected Improvement(EI) 를 사용
bo.maximize(init_points=4, n_iter=50, acq='ei')

# ‘iter’는 반복 회차, ‘target’은 목적 함수의 값, 나머지는 입력값을 나타냅니다. 
# 현재 회차 이전까지 조사된 함숫값들과 비교하여, 현재 회차에 최댓값이 얻어진 경우, 
# bayesian-optimization 라이브러리는 이를 자동으로 다른 색 글자로 표시하는 것을 확인

# 찾은 파라미터 값 확인
print('===============')
print(bo.max)
print('===============')
print("bo fit 소요 시간: %.3fms" %((time.time() - start_time)*1000) )

'''


from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'num_leaves':np.array(range(16,1024,1)), # (16, 1024),
    'max_depth':np.array(range(7,20,1)), # (7, 20),
    'learning_rate':np.arange(0.001,0.1,0.01), # (0.001, 0.1),
    'n_estimators':np.array(range(50,200,1)), # (50, 200),
}
start_time = time.time()
RSCV = RandomizedSearchCV(LGBMClassifier(), 
                        parameters, cv=5,
                        random_state=RANDOM_STATE,
                        verbose=2)
RSCV.fit(x_train,y_train)
score = RSCV.score(x_test, y_test)
print("score:", score)
best_params = RSCV.best_params_
print("최적의 파라미터:", best_params)
print("RSCV fit 소요 시간: %.3fs" %((time.time() - start_time)) )

