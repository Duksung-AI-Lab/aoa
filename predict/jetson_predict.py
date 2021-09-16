from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import numpy as np
import time
from serial import Serial
import collections

model1 = load_model('lstm_reg.h5')
model2 = load_model('lstm_class.h5')

# ser = Serial('COM9', 9600)  # 윈도우
ser = Serial('/dev/ttyACM0', 115200) #라즈베리파이
p = collections.deque()
SIZE = 100

while True:
    if ser.readable():
        res = ser.readline()
        s = res.decode()[:len(res) - 1]

        a = s.split(',')
        try:
            # parsing
            pa0 = int(a[3].split()[-1])
            pa1 = int(a[4].split()[-1])
            # print(pa0, pa1)

        except IndexError:
            continue

        if len(p) < SIZE:
            p.append([pa0, pa1])
            continue
        else:
            p.popleft()
            p.append([pa0, pa1])
        # print(p)

        # MinMaxScale (0~1)
        scaler = MinMaxScaler()
        data = scaler.fit_transform(p)
        data = np.reshape(data, (1, SIZE, 2))
        # print(data.shape)

        # start = time.time()
        # pred1 = model1.predict(data)
        # print('regression time: ', (time.time() - start), 'micros')

        start = time.time()
        pred2 = np.argmax(model2.predict(data), axis=-1)
        print('classification time: ', (time.time() - start), 'micros')

        # # print(pred1.reshape(1, len(pred1))[0])
        # # print(pred1.tolist()[0])
        # result1 = scaler.inverse_transform(pred1)
        # # result1 = list(map(lambda x: x * 180 - 90, pred1))
        # print(result1)
        result2 = list(map(lambda x: x * 10 - 90, pred2))
        print(result2, a[0].split()[-1])
