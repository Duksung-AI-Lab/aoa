import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import LSTM

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
#
# np.random.seed(0)
# tf.set_random_seed(0)
tf.random.set_seed(0)

train = pd.read_csv('../dataset/data_split/aoa_train.csv', engine='python')
val = pd.read_csv('../dataset/data_split/aoa_val.csv', engine='python')
test = pd.read_csv('../dataset/data_split/aoa_test.csv', engine='python')

train.columns = ['index', 'pa0', 'pa1', 'angle']
train = train.drop(['index'], axis=1)

val.columns = ['index', 'pa0', 'pa1', 'angle']
val = val.drop(['index'], axis=1)

test.columns = ['index', 'pa0', 'pa1', 'angle']
test = test.drop(['index'], axis=1)

# print(train.head())
print(train.shape, val.shape, test.shape)

'''0~18 scaling for one-hot encoding'''
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

'''feature/label split'''
train_feature = train[feature_cols]
train_label = train[label_cols]

val_feature = val[feature_cols]
val_label = val[label_cols]

test_feature = test[feature_cols]
test_label = test[label_cols]

'''scaling'''
scaler = MaxAbsScaler()
scaler.fit(train_feature)

train_feature = scaler.transform(train_feature)
val_feature = scaler.transform(val_feature)
test_feature = scaler.transform(test_feature)

# 객체를 pickled binary file 형태로 저장한다
file_name = '../model/aoa_cls_mas.pkl'
joblib.dump(scaler, file_name)

# make dataset 함수에 넣기 위해 data frame 형식으로 변환
train_feature = pd.DataFrame(train_feature)
train_feature.columns = feature_cols

val_feature = pd.DataFrame(val_feature)
val_feature.columns = feature_cols

test_feature = pd.DataFrame(test_feature)
test_feature.columns = feature_cols

'''make dataset'''
x_train, y_train = make_dataset(train_feature, train_label)
x_valid, y_valid = make_dataset(val_feature, val_label)
x_test, y_test = make_dataset(test_feature, test_label)

'''one-hot encoding'''
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

print(model.summary())

history = model.fit(x_train, y_train,
                    epochs=20,
                    validation_data=(x_valid, y_valid),
                    callbacks=[early_stop],
                    verbose=2)

print("\n Test Accuracy: %.4f" % (model.evaluate(x_test, y_test))[1])

y_pred = model.predict(x_test)

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

model.save('../model/LSTM_class.h5')
