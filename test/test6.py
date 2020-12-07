import numpy as np
import pickle

file_name = './test/model/temp_model.pkl'
model = pickle.load(open(file_name, "rb"))


x_predict = np.load('./test/data/mangi_predict_data.npy')
x_predict = x_predict.reshape(1, x_predict.shape[0])
print(x_predict.shape)

y_predict = model.predict(x_predict)
print(y_predict)