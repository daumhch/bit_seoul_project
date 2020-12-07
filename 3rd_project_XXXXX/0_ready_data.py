import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 디버그 메시지 끄기


import tensorflow as tf

reader = tf.data.TFRecordDataset(filenames='./3rd_project_XXXXX/nsynth-test.tfrecord')
print(reader)