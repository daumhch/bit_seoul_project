import numpy as np
import matplotlib.pyplot as plt

test_wav_filename = np.load('./test/data/test_wav_filename.npy')
test_wav_data = np.load('./test/data/test_wav_data.npy')
test_wav_target = np.load('./test/data/test_wav_target.npy')

print("test_wav_data.shape:", test_wav_data.shape)
print("test_wav_target.shape:", test_wav_target.shape)


'''
# index = 555
# print("test_wav_filename:",test_wav_filename[index])
# print("test_wav_target:",test_wav_target[index])
# plt.plot(test_wav_data[index])
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.title("Power spectrum")
# plt.show()
'''
from sklearn.model_selection import train_test_split 
x_train,x_test, y_train,y_test = train_test_split(
    test_wav_data, test_wav_target, train_size=0.8, test_size=0.2)

print("trian_test_split")
print("x_train.shape",x_train.shape)
print("x_test.shape",x_test.shape)


from xgboost import XGBClassifier

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("score:",score)

# score: 0.8585365853658536

# npy로 만든 파일을 불러와서 모델을 돌려 예측한다

import pickle
file_name = './test/model/temp_model.pkl'
pickle.dump(model, open(file_name, "wb"))


