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

SIZE=20

def make_dataset(data, label, window_size=SIZE):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array((label.iloc[i + window_size])))

    return np.array(feature_list), np.array(label_list)

def get_scaler(df):
    x_scaler = RobustScaler()
    ##학습데이터를 기준으로 scale
    df.columns = ['dummy', 'pa0', 'pa1', 'angle']
    df = df.drop(['dummy'], axis=1)
    x_scaler.fit(df[['pa0', 'pa1']])


    return x_scaler

def train_test_val(df, x_scaler):

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

    for i in df_label:
        df_label[i] += 90
        df_label[i] //= 10

    print(df_label)
    # dataset 생성
    df_x, df_y = make_dataset(df_feature, df_label)

    print(df_y.shape)
    return df_x, df_y
    # dataset 생성

with tf.device('/gpu:0'):
    np.random.seed(0)
    tf.set_random_seed(0)

    df_train = pd.read_csv('./dataset/aoa_train.csv', engine='python')
    df_val = pd.read_csv('./dataset/aoa_val.csv', engine='python')
    df_test = pd.read_csv('./dataset/aoa_test.csv', engine='python')

    x_scaler=get_scaler(df_train)

    train_x, train_y = train_test_val(df_train, x_scaler)
    valid_x, valid_y =  train_test_val(df_val, x_scaler)
    test_x, test_y =  train_test_val(df_test, x_scaler)

    train_x = train_x.reshape(-1, SIZE, 1, 2)
    valid_x = valid_x.reshape(-1, SIZE, 1, 2)
    test_x = test_x.reshape(-1, SIZE, 1, 2)

    print(test_y)

    train_y = np_utils.to_categorical(train_y, num_classes=19)
    valid_y = np_utils.to_categorical(valid_y,num_classes=19)
    test_y = np_utils.to_categorical(test_y,num_classes=19)

    print(test_y)
    #print(train_x.shape, train_y.shape)
    # (227992, 20, 1, 2) (227992, 19)
    #print(valid_x.shape, valid_y.shape)
    # (75998, 20, 1, 2) (75998, 19)
    #print(test_x.shape, test_y.shape)
    # (75998, 20, 1, 2)(75998, 19)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(SIZE, 1, 2)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(19, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(train_x, train_y,
                        epochs=5,
                        validation_data=(valid_x, valid_y),
                        callbacks=[early_stop],
                        verbose=1)

    model.save('210916_rnn_cl.h5')
    print("\n Test " , (model.evaluate(test_x, test_y)))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, acc, marker='.', c="red", label='Train_acc')
    plt.plot(x_len, val_acc, marker='.', c="blue", label='val_acc')

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.show()

    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_vloss, marker='.', c="red", label='val loss')
    plt.plot(x_len, y_loss, marker='.', c="blue", label='loss')

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss/acc')
    plt.show()