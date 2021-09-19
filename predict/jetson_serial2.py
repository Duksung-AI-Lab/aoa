from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
import time
from serial import Serial
import collections

model1 = load_model('./model/210916_rnn_reg.h5')
#model2 = load_model('./model/LSTM_class_2.h5')
#model1.summary()

df_train=pd.read_csv('./dataset/aoa_train.csv', engine='python')
scaler = RobustScaler()
df_train.columns = ['dummy', 'pa0', 'pa1', 'angle']
scaler.fit(df_train[['pa0','pa1']])

angle_scaler = MinMaxScaler()
angle_scaler.fit(df_train[['angle']])
#plot_model(model1, to_file='./model/210903_rnn_reg_model.png', show_shapes=True)

# ser = Serial('COM9', 9600)  # 윈도우
ser = Serial('/dev/ttyACM0', 115200) #라즈베리파이
p = collections.deque()
SIZE = 20
#label=[i for i in range(-90,91,10)]
while True:
    if ser.readable():
        res = ser.readline()
        s = res.decode()[:len(res) - 1]

        a = s.split(',')
        try:
            # parsing
            pa0 = int(a[3].split()[-1])
            pa1 = int(a[4].split()[-1])

        except IndexError:
            continue

        if len(p) < SIZE:
            p.append([pa0, pa1])
            continue
        else:
            p.popleft()
            p.append([pa0, pa1])
        # print(p)

        # RobustScale
        data = scaler.transform(p)
        data = np.reshape(data, (-1, SIZE, 1, 2))
        #print(data.shape)

        # start = time.time()
        pred1 = model1.predict(data)
        # print('regression time: ', (time.time() - start), 'micros')

        aoa = int(a[0].split()[-1]) + 45
        #start = time.time()
        #pred2 = np.argmax(model2.predict(data), axis=-1)
        # print(pred1.reshape(1, len(pred1))[0])
        # result1 = list(map(lambda x: x * 180 - 90, pred1))


        #result1 = pred1[0][0]*180-90
        result1 = angle_scaler.inverse_transform(pred1)
        print('R:',int(result1), 'aoa : ', aoa)
