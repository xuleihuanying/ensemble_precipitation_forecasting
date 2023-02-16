
# conv lstm
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from numpy import concatenate
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler

data = np.loadtxt('D:/干旱/data/result8/qm_predict.txt') #month *[pts*lead*model]
row = 65
col = 37
pts = row*col
month = data.shape[0]
lead = 9
model_num = 8

index_nan = np.where(np.isnan(data))
data[index_nan] = 0

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

data = np.reshape(data_scaled, (month, pts, lead, model_num), order='F')
# (samples, time, rows, cols, channels)
data = np.reshape(data, (month, row, col, lead, model_num), order='F')
# data = np.transpose(data, (0, 3, 1, 2, 4))  # (n_samples, time, rows, cols, channels)

start_year = 1982
end_year = 2016
train_end_year = 2010
train_end = (train_end_year - start_year + 1)*12
# all = (end_year - start_year + 1)*12
year = (end_year - start_year + 1)

# It takes input of shape (samples,time_window_inputs,rows,cols,channels=1)
# and output shape is (samples,time_window_predictions=1,rows,cols,channels=1)
gpcc_china = np.loadtxt('D:/干旱/data/result8/gpcc_china.txt') #month*pts
index_nan = np.where(np.isnan(gpcc_china))
gpcc_china[index_nan] = 0
gpcc_scaled = scaler.fit_transform(gpcc_china)
gpcc_scaled = np.reshape(gpcc_scaled, (month, row, col), order='F')

lstm_predict = np.zeros((month, pts, lead), dtype=np.float)
for i in range(lead):
    obs = gpcc_scaled[:, :, :]
    obs = np.reshape(obs, (month, row, col, 1), order='F')
    model = data[:, :, :, i, :]
    # (samples, time, rows, cols, channels)
    model = np.reshape(model, (month, 1, row, col, model_num), order='F')
    train_x = model[0:train_end, :, :, :, :]
    train_y = obs[0:train_end, :, :, :]
    test_x = model

    # if sum(train_y) == 0:
    #     continue

    # # reshape input to be 3D [samples, timesteps, features]
    # train_x = train_x.reshape(( train_end, 1, model_num), order='F')
    # train_y = train_y.reshape(( train_end, 1), order='F')
    # test_x = test_x.reshape(( month, 1, model_num), order='F')

    # model = Sequential()
    # model.add(LSTM(10, input_shape=(train_x.shape[1], train_x.shape[2])))  # 50
    # model.add(Dense(1))
    # model.compile(loss='mae', optimizer='adam')

    model = Sequential()
    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                       input_shape=(1, row, col, model_num),
                       padding='same', return_sequences=True))
    # seq.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                      padding='same', return_sequences=True))
    # seq.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                         padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                         padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                         padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                         padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                         padding='same', return_sequences=True))

    model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
                       padding='same', data_format='channels_last', return_sequences=False)) #
    # seq.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.compile(loss='mse', optimizer='adam')

    # fit network
    history = model.fit(train_x, train_y, batch_size=train_x.shape[0],
            epochs=200, validation_split=0.1)
    # history = model.fit(train_x, train_y, epochs=100, \
    #                     batch_size = 100, verbose=2, shuffle=False) #train_x.shape[0]
    y_pre = model.predict(test_x)
    y_pre = np.reshape(y_pre, (month, pts), order='F')
    lstm_predict[:, :, i] = scaler.inverse_transform(y_pre)
    print ("%d"%(i))

# inverse transform
# for i in range(lead):
#     lstm_predict[:, :, i] = scaler.inverse_transform(lstm_predict[:, :, i])

lstm_predict = lstm_predict.reshape((month, pts*lead), order='F')
np.savetxt('D:/干旱/data/result8/conv_lstm_predict.txt', lstm_predict)


