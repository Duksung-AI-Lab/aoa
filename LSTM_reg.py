import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn import utils
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from tensorflow import optimizers
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(0)
tf.set_random_seed(0)

df = pd.read_csv('aoa_dataset.csv', engine='python')
df.columns = ['aoa', 'rssi0', 'rssi1', 'pa0', 'pa1', 'angle']
df = df.drop(['aoa', 'rssi0', 'rssi1'], axis=1)
print(df.head())
print(len(df))


def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


feature_cols = ['pa0', 'pa1']
label_cols = ['angle']

# feature/label split
df_feature = df[feature_cols]
df_label = df[label_cols]

# MinMaxScale (0~1)
scaler = MinMaxScaler()

df_feature = scaler.fit_transform(df_feature)
df_feature = pd.DataFrame(df_feature)
df_feature.columns = feature_cols

df_label = scaler.fit_transform(df_label)
df_label = pd.DataFrame(df_label)
df_label.columns = label_cols
print(df_label.head())

df_x, df_y = np.empty((0, 20, 2)), np.empty((0, 1))

for i in range(-90, 91, 10):
    # print(d1)

    # plt.figure()
    # plt.title(i)
    # plt.plot(d1['a1'][:], label='a1')
    # plt.plot(d1['a2'][:], label='a2')
    # plt.legend()
    # plt.show()

    d1_feature = df_feature[df['angle']==i]
    d1_label = df_label[df['angle']==i]

    # train dataset 생성
    d1_x, d1_y = make_dataset(d1_feature, d1_label)
    df_x = np.concatenate([df_x, d1_x])
    df_y = np.concatenate([df_y, d1_y])

print(df_x)
print(df_y)

# train, validation set
x_train, x_valid, y_train, y_valid = train_test_split(df_x, df_y, test_size=0.2, shuffle=True)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

print(x_train.shape, y_train.shape)
# (8152, 20, 2) (8152, 1)
print(x_valid.shape, y_valid.shape)
# (2548, 20, 2) (2548, 1)
print(x_test.shape, y_test.shape)
# (2039, 20, 2) (2039, 1)

model = Sequential()
model.add(LSTM(16,
               input_shape=(df_x.shape[1], df_x.shape[2]),
               activation='tanh',
               return_sequences=False))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(x_train, y_train,
                    epochs=50,
                    validation_data=(x_valid, y_valid),
                    callbacks=[early_stop],
                    verbose=1)

print("\n Test Loss: %.4f" % (model.evaluate(x_test, y_test)))

y_pred = model.predict(x_test)

# # inverse scale
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

plt.figure()
plt.plot(y_test, label='actual')
plt.plot(y_pred, label='prediction')
plt.legend()
plt.show()
