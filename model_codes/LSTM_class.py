import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(0)
tf.set_random_seed(0)

train = pd.read_csv('aoa_train.csv', engine='python')
val = pd.read_csv('aoa_val.csv', engine='python')
test = pd.read_csv('aoa_test.csv', engine='python')
train.columns = ['index', 'pa0', 'pa1', 'angle']
train = train.drop(['index'], axis=1)
val.columns = ['index', 'pa0', 'pa1', 'angle']
val = val.drop(['index'], axis=1)
test.columns = ['index', 'pa0', 'pa1', 'angle']
test = test.drop(['index'], axis=1)
print(train.head())
print(len(train), len(val), len(test))

# one-hot encoding
for d in (train, val, test):
    for i in range(len(d)):
        d['angle'][i] += 90
        d['angle'][i] //= 10
    print(d.head())


def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)


feature_cols = ['pa0', 'pa1']
label_cols = ['angle']
# train_x, train_y = np.empty((0, 20, 2)), np.empty((0, 1))


def data(d1):
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
    return d1_x, d1_y


x_train, y_train = data(train)
x_valid, y_valid = data(val)
x_test, y_test = data(test)

# one-hot encoding
y_train = np_utils.to_categorical(y_train, num_classes=19)
y_valid = np_utils.to_categorical(y_valid, num_classes=19)
y_test = np_utils.to_categorical(y_test, num_classes=19)

print(x_train.shape, y_train.shape)
# (303992, 20, 2) (303992, 19)
print(x_valid.shape, y_valid.shape)
# (37998, 20, 2) (37998, 19)
print(x_test.shape, y_test.shape)
# (37998, 20, 2) (37998, 19)

model = Sequential()
model.add(LSTM(16,
               input_shape=(x_train.shape[1], x_train.shape[2]),
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
plt.ylabel('acc')
plt.show()

# model.save('model/LSTM_class.h5')
