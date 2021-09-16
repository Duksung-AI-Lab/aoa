from tensorflow.keras.models import load_model
#from keras.models import load_model
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

model1 = load_model('lstm_reg.h5')
model2 = load_model('lstm_class.h5')

# data1 = np.load('dataset/aoa/data_reg.npy')
# data2 = np.load('dataset/aoa/data_class.npy')

data1 = np.load('data_reg.npy')
data2 = np.load('data_class.npy')

start = time.time()
pred1 = model1.predict(data1)
print(pred1)
print('regression time: ', (time.time() - start)/len(data1), 'micros')

start = time.time()
pred2 = np.argmax(model2.predict(data2), axis=-1)
print(pred2)
print('classification time: ', (time.time() - start)/len(data2), 'micros')

# print(pred1.reshape(1, len(pred1))[0])
# print(pred1.tolist()[0])
# result1 = list(map(lambda x: x*180-90, pred1))
# print(result1)
# result2 = list(map(lambda x: x*15-90, pred2))
# print(result2)
