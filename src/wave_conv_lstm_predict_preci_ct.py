
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

all_climate_index = np.loadtxt('D:/干旱/data/result8/all_climate_index.txt') # 684*4
scaler_2 = MinMaxScaler(feature_range=(0, 1))
all_climate_index = scaler_2.fit_transform(all_climate_index)
ci_col = all_climate_index.shape[1]
train_data_ci = all_climate_index[0:train_end, :]
all_ci_pts = np.zeros((month, pts, ci_col), dtype=np.float)
for i in range(pts):
    all_ci_pts[:, i, :] = all_climate_index
all_ci_pts = np.reshape(all_ci_pts, (month, row, col, ci_col), order='F')

tem = np.loadtxt('D:/干旱/data/result8/cruts_tem_6016.txt') # 684*2405
index_nan = np.where(np.isnan(tem))
tem[index_nan] = 0
scaler_3 = MinMaxScaler(feature_range=(0, 1))
tem = scaler_3.fit_transform(tem)
tem = np.reshape(tem, (month, row, col), order='F')

tem_min = np.loadtxt('D:/干旱/data/result8/cruts_tem_min_6016.txt')
index_nan = np.where(np.isnan(tem_min))
tem_min[index_nan] = 0
scaler_4 = MinMaxScaler(feature_range=(0, 1))
tem_min = scaler_4.fit_transform(tem_min)
tem_min = np.reshape(tem_min, (month, row, col), order='F')

tem_max = np.loadtxt('D:/干旱/data/result8/cruts_tem_max_6016.txt')
index_nan = np.where(np.isnan(tem_max))
tem_max[index_nan] = 0
scaler_5 = MinMaxScaler(feature_range=(0, 1))
tem_max = scaler_5.fit_transform(tem_max)
tem_max = np.reshape(tem_max, (month, row, col), order='F')


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
    
    # x1 = np.concatenate([np.zeros((i + 1, row, col), dtype=np.float), preci[0:all_end - i - 1, :, :]], axis=0)  # 提前1个月降雨
    x4 = np.concatenate([np.zeros((i + 1, row, col), dtype=np.float), tem[0:all_end - i - 1, :, :]], axis=0)  # 提前1个月降雨
    x5 = np.concatenate([np.zeros((i + 1, row, col), dtype=np.float), tem_min[0:all_end - i - 1, :, :]], axis=0)  # 提前1个月降雨
    x6 = np.concatenate([np.zeros((i + 1, row, col), dtype=np.float), tem_max[0:all_end - i - 1, :, :]], axis=0)  # 提前1个月降雨
    x7 = np.concatenate([np.zeros((i + 1, row, col, ci_col), dtype=np.float), all_ci_pts[0:all_end - i - 1, :, :, :]], axis=0)

    y = preci[0:all_end, :, :]

    # x1 = np.reshape(x1, (all_end, row, col, 1), order='F')
    x4 = np.reshape(x4, (all_end, row, col, 1), order='F')
    x5 = np.reshape(x5, (all_end, row, col, 1), order='F')
    x6 = np.reshape(x6, (all_end, row, col, 1), order='F')
    x7 = np.reshape(x7, (all_end, row, col, ci_col), order='F')
    train_sub = np.concatenate([x1, x2, x3, x4, x5, x6, x7], axis=3)

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
np.savetxt('D:/干旱/data/result8/wave_conv_lstm_predict_p_ct.txt', lstm_predict)


