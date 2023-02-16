
# lstm
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

start_year = 1982
end_year = 2016
train_end_year = 2010
train_end = (train_end_year - start_year + 1)*12
# all = (end_year - start_year + 1)*12
year = (end_year - start_year + 1)

gpcc_china = np.loadtxt('D:/干旱/data/result8/gpcc_china.txt') #month*pts
index_nan = np.where(np.isnan(gpcc_china))
gpcc_china[index_nan] = 0
gpcc_scaled = scaler.fit_transform(gpcc_china)

lstm_predict = np.zeros((month, pts, lead), dtype=np.float)
for i in range(pts):
    for j in range(lead):
        obs = gpcc_scaled[:, i]
        model = data[:, i, j, :]
        model = np.reshape(model, (month, model_num), order='F')
        train_x = model[0:train_end, :]
        train_y = obs[0:train_end]
        test_x = model

        if sum(train_y) == 0:
            continue

        # reshape input to be 3D [samples, timesteps, features]
        train_x = train_x.reshape(( train_end, 1, model_num), order='F')
        train_y = train_y.reshape(( train_end, 1), order='F')
        test_x = test_x.reshape(( month, 1, model_num), order='F')

        model = Sequential()
        model.add(LSTM(10, input_shape=(train_x.shape[1], train_x.shape[2])))  # 50
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

lstm_predict = lstm_predict.reshape((month,pts*lead), order='F')
np.savetxt('D:/干旱/data/result8/lstm_predict.txt', lstm_predict)


