import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.models import load_model

df = pd.read_csv('../dataset/dataset_211001.csv', engine='python')

df.columns = ['aoa', 'rssi0', 'rssi1', 'pa0', 'pa1', 'angle']
df = df.drop(['aoa', 'rssi0', 'rssi1'], axis=1)

print("data shape", df.shape)
# (190012, 3)

'''0~18 scaling for one-hot encoding'''
for i in range(len(df)):
    df['angle'][i] += 90
    df['angle'][i] //= 10
print(df[-5:])


def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


'''feature/label split'''
feature_cols = ['pa0', 'pa1']
label_cols = ['angle']

test_feature = df[feature_cols]
test_label = df[label_cols]

'''scaling'''
# pickled binary file 형태로 저장된 객체를 로딩한다
# file_name = '../model/aoa_reg_mas.pkl'
file_name = '../model/aoa_cls_mas.pkl'
scaler = joblib.load(file_name)

test_feature = scaler.transform(test_feature)

# make dataset 함수에 넣기 위해 data frame 형식으로 변환
test_feature = pd.DataFrame(test_feature)
test_feature.columns = feature_cols

'''make dataset'''
x_test, y_test = make_dataset(test_feature, test_label)

'''one-hot encoding'''
y_test_encoded = np_utils.to_categorical(y_test, num_classes=19)

print(x_test.shape, y_test_encoded.shape)
# (189992, 20, 2) (189992, 1) reg
# (189992, 20, 2) (189992, 19) class

# model = load_model('../model/LSTM_reg.h5')
model = load_model('../model/LSTM_class.h5')

y_pred = model.predict(x_test)
print(y_pred.shape)

# print("\n Test Loss: %.4f" % (model.evaluate(x_test, y_test)))
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test_encoded))[1])
y_pred = np.argmax(y_pred, axis=1)
print(y_pred, y_pred.shape)

plt.figure()
plt.plot(y_test, label='actual')
plt.plot(y_pred, label='prediction')
plt.legend()
plt.show()
