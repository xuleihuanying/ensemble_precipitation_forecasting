
# lstm, temperature & climate index, wavelet
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from numpy import concatenate
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
preci = np.loadtxt('D:/干旱/data/result8/gpcc_china_6016.txt') #month *[pts],684*2405

index_nan = np.where(np.isnan(preci))
preci[index_nan] = 0

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
preci = scaler.fit_transform(preci)


start_year = 1960
end_year = 2016
train_end_year = 2010
year = (end_year - start_year + 1)

train_end = (train_end_year - start_year + 1) * 12
all_end = (end_year - start_year + 1) * 12

all_climate_index = np.loadtxt('D:/干旱/data/result8/all_climate_index.txt')
scaler_2 = MinMaxScaler(feature_range=(0, 1))
all_climate_index = scaler_2.fit_transform(all_climate_index)
ci_col = all_climate_index.shape[1]
train_data_ci = all_climate_index[0:train_end, :]

# tem = np.loadtxt('D:/干旱/data/result8/cruts_tem_6016.txt')
# tem_min = np.loadtxt('D:/干旱/data/result8/cruts_tem_min_6016.txt')
# tem_max = np.loadtxt('D:/干旱/data/result8/cruts_tem_max_6016.txt')

tem = np.loadtxt('D:/干旱/data/result8/cruts_tem_6016.txt') # 684*2405
index_nan = np.where(np.isnan(tem))
tem[index_nan] = 0
scaler_3 = MinMaxScaler(feature_range=(0, 1))
tem = scaler_3.fit_transform(tem)
# tem = np.reshape(tem, (month, row, col), order='F')

tem_min = np.loadtxt('D:/干旱/data/result8/cruts_tem_min_6016.txt')
index_nan = np.where(np.isnan(tem_min))
tem_min[index_nan] = 0
scaler_4 = MinMaxScaler(feature_range=(0, 1))
tem_min = scaler_4.fit_transform(tem_min)
# tem_min = np.reshape(tem_min, (month, row, col), order='F')

tem_max = np.loadtxt('D:/干旱/data/result8/cruts_tem_max_6016.txt')
index_nan = np.where(np.isnan(tem_max))
tem_max[index_nan] = 0
scaler_5 = MinMaxScaler(feature_range=(0, 1))
tem_max = scaler_5.fit_transform(tem_max)
# tem_max = np.reshape(tem_max, (month, row, col), order='F')

train_label_data = preci[0:train_end, :]
m = train_label_data.shape[0]
n = train_label_data.shape[1]

lead = 6
m2 = preci.shape[0]
n2 = preci.shape[1]
month = preci.shape[0]

preci_wavelet = np.loadtxt('D:/干旱/data/result8/preci_wavelet.txt') #684*(2405*4)
preci_wavelet = np.reshape(preci_wavelet, (m2,n2,4), order='F')
wave_col = preci_wavelet.shape[2]
scaler_2 = MinMaxScaler(feature_range=(0, 1))
for i in range(4):
    preci_wavelet[:,:,i] = scaler_2.fit_transform(preci_wavelet[:,:,i])

lstm_predict = np.zeros((m2, n2, lead), dtype=np.float)

# data = np.reshape(data_scaled, (month, pts, lead, model_num), order='F')
for i in range(n):
    for j in range(lead):
        s = preci_wavelet[0:train_end - j - 3, i, :]
        s = np.reshape(s, (s.shape[0], s.shape[1]))
        x1 = np.concatenate([np.zeros((j + 3, wave_col), dtype=np.float), s], axis=0)  # 提前1个月降雨
        
        s = preci_wavelet[0:train_end - j - 2, i, :]
        s = np.reshape(s, (s.shape[0], s.shape[1]))
        x2 = np.concatenate([np.zeros((j + 2, wave_col), dtype=np.float), s], axis=0)  # 提前1个月降雨
        
        s = preci_wavelet[0:train_end - j - 1, i, :]
        s = np.reshape(s, (s.shape[0], s.shape[1]))
        x3 = np.concatenate([np.zeros((j + 1, wave_col), dtype=np.float), s], axis=0)  # 提前1个月降雨
        
        # x1 = np.concatenate([np.zeros((j + 1), dtype=np.float), preci[0:train_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x4 = np.concatenate([np.zeros((j + 1), dtype=np.float), tem[0:train_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x5 = np.concatenate([np.zeros((j + 1), dtype=np.float), tem_min[0:train_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x6 = np.concatenate([np.zeros((j + 1), dtype=np.float), tem_max[0:train_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x7 = np.concatenate([np.zeros((j + 1, ci_col), dtype=np.float), all_climate_index[0:train_end-j-1, :]], axis=0)

        # x1 = np.reshape(x1, (train_end,1), order='F')
        x4 = np.reshape(x4, (train_end, 1), order='F')
        x5 = np.reshape(x5, (train_end, 1), order='F')
        x6 = np.reshape(x6, (train_end, 1), order='F')
        x7 = np.reshape(x7, (train_end, ci_col), order='F')

        train_sub = np.concatenate([x1, x2, x3, x4, x5, x6, x7], axis=1)
        label_sub = train_label_data[0:, i]

        if sum(label_sub) == 0:
            continue

        s = preci_wavelet[0:all_end - j - 3, i, :]
        s = np.reshape(s, (s.shape[0], s.shape[1]))
        x1 = np.concatenate([np.zeros((j + 3, wave_col), dtype=np.float), s], axis=0)  # 提前1个月降雨
        
        s = preci_wavelet[0:all_end - j - 2, i, :]
        s = np.reshape(s, (s.shape[0], s.shape[1]))
        x2 = np.concatenate([np.zeros((j + 2, wave_col), dtype=np.float), s], axis=0)  # 提前1个月降雨
        
        s = preci_wavelet[0:all_end - j - 1, i, :]
        s = np.reshape(s, (s.shape[0], s.shape[1]))
        x3 = np.concatenate([np.zeros((j + 1, wave_col), dtype=np.float), s], axis=0)  # 提前1个月降雨
        
        # x1 = np.concatenate([np.zeros((j + 1), dtype=np.float), preci[0:all_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x4 = np.concatenate([np.zeros((j + 1), dtype=np.float), tem[0:all_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x5 = np.concatenate([np.zeros((j + 1), dtype=np.float), tem_min[0:all_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x6 = np.concatenate([np.zeros((j + 1), dtype=np.float), tem_max[0:all_end - j - 1, i]], axis=0)  # 提前1个月降雨
        x7 = np.concatenate([np.zeros((j + 1, ci_col), dtype=np.float), all_climate_index[0:all_end - j - 1, :]],axis=0)

        # x1 = np.reshape(x1, (all_end, 1), order='F')
        x4 = np.reshape(x4, (all_end, 1), order='F')
        x5 = np.reshape(x5, (all_end, 1), order='F')
        x6 = np.reshape(x6, (all_end, 1), order='F')
        x7 = np.reshape(x7, (all_end, ci_col), order='F')

        x_pre = np.concatenate([x1, x2, x3, x4, x5, x6, x7], axis=1)
        feature = train_sub.shape[1]
        # reshape input to be 3D [samples, timesteps, features]
        train_x = train_sub.reshape(( train_end, 1, feature), order='F')
        train_y = label_sub.reshape(( train_end, 1), order='F')
        test_x = x_pre.reshape(( all_end, 1, feature), order='F')

        model = Sequential()
        model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))  # 50
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        history = model.fit(train_x, train_y, epochs=1000, \
                            batch_size = train_x.shape[0], verbose=2, shuffle=False, validation_split=0.1) #train_x.shape[0]
        y_pre = model.predict(test_x)
        lstm_predict[:, i, j] = y_pre[:, 0]
        print ("%d %d"%(i,j))
        # destroy the current TF graph and create a new one
        K.clear_session()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))

# inverse transform
for i in range(lead):
    lstm_predict[:, :, i] = scaler.inverse_transform(lstm_predict[:, :, i])

lstm_predict = lstm_predict.reshape((m2, n2*lead), order='F')
np.savetxt('D:/干旱/data/result8/wave_lstm_predict_p_ct.txt', lstm_predict)

