import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn import utils
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping
from tensorflow import optimizers
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

SIZE = 20


def make_dataset(data, label, window_size=SIZE):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array(label.iloc[i + window_size]))
    return np.array(feature_list), np.array(label_list)


def get_scaler(df):
    x_scaler = RobustScaler()
    y_scaler = MinMaxScaler()
    ##학습데이터를 기준으로 scale
    df.columns = ['dummy', 'pa0', 'pa1', 'angle']
    df = df.drop(['dummy'], axis=1)
    x_scaler.fit(df[['pa0', 'pa1']])
    y_scaler.fit(df[['angle']])

    return x_scaler, y_scaler


def train_test_val(df, x_scaler, y_scaler):
    df_x, df_y = np.empty((0, SIZE, 2)), np.empty((0, 1))
    df.columns = ['dummy', 'pa0', 'pa1', 'angle']

    feature_cols = ['pa0', 'pa1']
    label_cols = ['angle']

    # feature/label split
    df_feature = df[feature_cols]
    df_label = df[label_cols]

    print(df_feature.head())
    df_feature = x_scaler.transform(df_feature)
    df_feature = pd.DataFrame(df_feature)
    df_feature.columns = feature_cols

    df_label = y_scaler.transform(df_label)
    df_label = pd.DataFrame(df_label)
    df_label.columns = label_cols
    print(df_feature.head())

    for i in range(-90, 91, 10):
        d1_feature = df_feature[df['angle'] == i]
        d1_label = df_label[df['angle'] == i]

        # dataset 생성
        d1_x, d1_y = make_dataset(d1_feature, d1_label)

        df_x = np.concatenate([df_x, d1_x])
        df_y = np.concatenate([df_y, d1_y])

    return df_x, df_y
    # dataset 생성


with tf.device('/gpu:1'):
    np.random.seed(0)
    tf.set_random_seed(0)

    df_train = pd.read_csv('./dataset/aoa_train.csv', engine='python')
    df_val = pd.read_csv('./dataset/aoa_val.csv', engine='python')
    df_test = pd.read_csv('./dataset/aoa_test.csv', engine='python')

    x_scaler, y_scaler = get_scaler(df_train)

    train_x, train_y = train_test_val(df_train, x_scaler, y_scaler)
    valid_x, valid_y = train_test_val(df_val, x_scaler, y_scaler)
    test_x, test_y = train_test_val(df_test, x_scaler, y_scaler)

    train_x = train_x.reshape(-1, SIZE, 1, 2)
    valid_x = valid_x.reshape(-1, SIZE, 1, 2)
    test_x = test_x.reshape(-1, SIZE, 1, 2)

    print(test_x)
    print(test_y)

    print(train_x.shape, train_y.shape)
    # (227632, 20)(227632, 1)
    print(valid_x.shape, valid_y.shape)
    # (75638, 20)(75638, 1)
    print(test_x.shape, test_y.shape)
    # (75638, 20)(75638, 1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(SIZE, 1, 2)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    print(train_x)
    print(train_y)
    history = model.fit(train_x, train_y,
                        epochs=50,
                        validation_data=(valid_x, valid_y),
                        callbacks=[early_stop],
                        verbose=1)

    print("\n Test Loss: %.4f" % (model.evaluate(test_x, test_y)))
    model.save('210916_rnn_reg2.h5')

    y_pred = model.predict(test_x)

    plt.plot(len(test_x))

    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c="red", label='val loss')
    plt.plot(x_len, y_loss, marker='.', c="blue", label='loss')

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.show()