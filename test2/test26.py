import numpy as np
import pickle

file_name = './test2/model/temp_model.pkl'
model = pickle.load(open(file_name, "rb"))



x_predict = np.load('./test2/data/test_predict_data.npy')
x_predict = x_predict.reshape(1, x_predict.shape[0])
print(x_predict.shape)

import pickle as pk
pca_reload = pk.load(open("./test2/model/pca.pkl",'rb'))
x_predict = pca_reload .transform(x_predict)

y_predict = model.predict(x_predict)
print(y_predict)







