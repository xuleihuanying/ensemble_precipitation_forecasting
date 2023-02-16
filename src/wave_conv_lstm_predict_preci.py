
# conv lstm, wave
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

preci = np.loadtxt('D:/干旱/data/result8/gpcc_china_6016.txt') #month *[pts],684*2405

index_nan = np.where(np.isnan(preci))
preci[index_nan] = 0

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
preci = scaler.fit_transform(preci)

row = 65
col = 37
pts = row*col
month = preci.shape[0]
lead = 6

# (samples, time, rows, cols, channels)
preci = np.reshape(preci, (month, row, col), order='F')
# data = np.transpose(data, (0, 3, 1, 2, 4))  # (n_samples, time, rows, cols, channels)

start_year = 1960
end_year = 2016
train_end_year = 2010
train_end = (train_end_year - start_year + 1)*12
all_end = (end_year - start_year + 1) * 12
# all = (end_year - start_year + 1)*12
year = (end_year - start_year + 1)

preci_wavelet = np.loadtxt('D:/干旱/data/result8/preci_wavelet.txt') #684*(2405*4)
preci_wavelet = np.reshape(preci_wavelet, (month, pts, 4), order='F')
wave_col = preci_wavelet.shape[2]
scaler_2 = MinMaxScaler(feature_range=(0, 1))
for i in range(4):
    preci_wavelet[:,:,i] = scaler_2.fit_transform(preci_wavelet[:,:,i])
preci_wavelet = np.reshape( preci_wavelet, (month, row, col, wave_col), order='F')

# It takes input of shape (samples,time_window_inputs,rows,cols,channels=1)
# and output shape is (samples,time_window_predictions=1,rows,cols,channels=1)

lstm_predict = np.zeros((month, pts, lead), dtype=np.float)
for i in range(lead):
    x1 = np.concatenate([np.zeros((i + 3, row, col, wave_col), dtype=np.float), preci_wavelet[0:all_end - i - 3, :, :, :]], axis=0)
    x2 = np.concatenate([np.zeros((i + 2, row, col, wave_col), dtype=np.float), preci_wavelet[0:all_end - i - 2, :, :, :]], axis=0)
    x3 = np.concatenate([np.zeros((i + 1, row, col, wave_col), dtype=np.float), preci_wavelet[0:all_end - i - 1, :, :, :]], axis=0)
    
    y = preci[0:all_end, :, :]

##    x1 = np.reshape(x1, (all_end, row, col, 1), order='F')
##    x2 = np.reshape(x2, (all_end, row, col, 1), order='F')
##    x3 = np.reshape(x3, (all_end, row, col, 1), order='F')
##    x4 = np.reshape(x4, (all_end, row, col, 1), order='F')
##    x5 = np.reshape(x5, (all_end, row, col, 1), order='F')
##    x6 = np.reshape(x6, (all_end, row, col, 1), order='F')
    train_sub = np.concatenate([x1, x2, x3], axis=3) #month*row*col*time
    # train_sub = np.transpose(train_sub, (0, 3, 1, 2))#month*time*row*col

    y = np.reshape(y, (month, row, col, 1), order='F')
    # (samples, time, rows, cols, channels)
    train_sub = np.reshape(train_sub, (month, 1, row, col, train_sub.shape[3]), order='F')
    train_x = train_sub[0:train_end, :, :, :, :]
    train_y = y[0:train_end, :, :, :]
    test_x = train_sub

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
                       input_shape=(1, row, col, train_sub.shape[4]),
                       padding='same', return_sequences=True))
    # seq.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=8, kernel_size=(3, 3),
                      padding='same', return_sequences=True))
    # seq.add(BatchNormalization())

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
    lstm_predict[:, :, i] = y_pre
    # lstm_predict[:, :, i] = scaler.inverse_transform(y_pre)
    print ("%d"%(i))

# inverse transform
for i in range(lead):
    lstm_predict[:, :, i] = scaler.inverse_transform(lstm_predict[:, :, i])

lstm_predict = lstm_predict.reshape((month, pts*lead), order='F')
np.savetxt('D:/干旱/data/result8/wave_conv_lstm_predict_p.txt', lstm_predict)


