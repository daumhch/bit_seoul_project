import numpy as np
import matplotlib.pyplot as plt

test_wav_filename = np.load('./test2/data/test_wav_filename.npy')
test_wav_data = np.load('./test2/data/test_wav_data.npy')
test_wav_target = np.load('./test2/data/test_wav_target.npy')

print("test_wav_data.shape:", test_wav_data.shape)
print("test_wav_target.shape:", test_wav_target.shape)



# 1.5 PCA
from sklearn.decomposition import PCA
cumsum_standard = 0.95
pca = PCA()
pca.fit(test_wav_data)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= cumsum_standard) +1
print("n_components:",d) # 154

pca = PCA(n_components=0.95)
test_wav_data = pca.fit_transform(test_wav_data)
print("after test_wav_data.shape:", test_wav_data.shape)

import pickle as pk
pk.dump(pca, open("./test2/model/pca.pkl","wb"))
