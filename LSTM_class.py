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

# one-hot encoding
for i in range(len(df)):
    df['angle'][i] += 90
    df['angle'][i] //= 10
# print(df.head())

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


feature_cols = ['pa0', 'pa1']
label_cols = ['angle']
df_x, df_y = np.empty((0, 20, 2)), np.empty((0, 1))

for i in range(13):
    d1 = df[df['angle']==i]
    # print(d1)

    # plt.figure()
    # plt.title(i)
    # plt.plot(d1['a1'][:], label='a1')
    # plt.plot(d1['a2'][:], label='a2')
    # plt.legend()
    # plt.show()

    # feature/label split
    d1_feature = d1[feature_cols]
    d1_label = d1[label_cols]

    # MinMaxScale (0~1)
    scaler = MinMaxScaler()
    d1_feature = scaler.fit_transform(d1_feature)
    d1_feature = pd.DataFrame(d1_feature)
    d1_feature.columns = feature_cols

    # train dataset 생성
    d1_x, d1_y = make_dataset(d1_feature, d1_label)
    print(i, ":", len(d1), len(d1_x))
    df_x = np.concatenate([df_x, d1_x])
    df_y = np.concatenate([df_y, d1_y])

# print(df_x)
# print(df_y)

# train, validation set
x_train, x_valid, y_train, y_valid = train_test_split(df_x, df_y, test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

# one-hot encoding
y_train = np_utils.to_categorical(y_train, num_classes=19)
y_valid = np_utils.to_categorical(y_valid, num_classes=19)
y_test = np_utils.to_categorical(y_test, num_classes=19)

print(x_train.shape, y_train.shape)
# (8152, 20, 2) (8152, 13)
print(x_valid.shape, y_valid.shape)
# (2548, 20, 2) (2548, 13)
print(x_test.shape, y_test.shape)
# (2039, 20, 2) (2039, 13)

model = Sequential()
model.add(LSTM(16,
               input_shape=(df_x.shape[1], df_x.shape[2]),
               activation='tanh',
               return_sequences=False))
model.add(Dense(19, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(x_train, y_train,
                    epochs=20,
                    validation_data=(x_valid, y_valid),
                    callbacks=[early_stop],
                    verbose=1)

print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test))[1])

y_pred = model.predict(x_test)

# # inverse scale
# y_pred = scaler.inverse_transform(y_pred)
# y_test = scaler.inverse_transform(y_test)

acc = history.history['acc']
val_acc = history.history['val_acc']
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
plt.plot(x_len, val_acc, marker='.', c="blue", label='Test_acc')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss/acc')
plt.show()
