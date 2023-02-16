% bma multimodel 
clear all
clc
linear_p = importdata('D:\干旱\data\result8\linear_predict_p.mat');%684*2405*6
linear_p_ct = importdata('D:\干旱\data\result8\linear_predict_p_ct.mat');
[m1 m2 m3] = size(linear_p);

svm_p = importdata('D:\干旱\data\result8\svm_predict_p.mat');
svm_p_ct = importdata('D:\干旱\data\result8\svm_predict_p_ct.mat');

ann_p = importdata('D:\干旱\data\result8\ann_predict_p.mat');
ann_p_ct = importdata('D:\干旱\data\result8\ann_predict_p_ct.mat');

rf_p = importdata('D:\干旱\data\result8\rf_predict_p.mat');
rf_p_ct = importdata('D:\干旱\data\result8\rf_predict_p_ct.mat');

lstm_p = load('D:\干旱\data\result8\lstm_predict_p.txt');%684*(2405*6)
lstm_p = reshape(lstm_p, [m1 m2 m3]);
lstm_p_ct = importdata('D:\干旱\data\result8\lstm_predict_p_ct.txt');
lstm_p_ct = reshape(lstm_p_ct, [m1 m2 m3]);

conv_lstm_p = load('D:\干旱\data\result8\conv_lstm_predict_p.txt');
conv_lstm_p = reshape(conv_lstm_p, [m1 m2 m3]);
conv_lstm_p_ct = importdata('D:\干旱\data\result8\conv_lstm_predict_p_ct.txt');
conv_lstm_p_ct = reshape(conv_lstm_p_ct, [m1 m2 m3]);

wave_linear_p = importdata('D:\干旱\data\result8\wave_linear_predict_p.mat');
wave_linear_p_ct = importdata('D:\干旱\data\result8\wave_linear_predict_p_ct.mat');

wave_svm_p = importdata('D:\干旱\data\result8\wave_svm_predict_p.mat');
wave_svm_p_ct = importdata('D:\干旱\data\result8\wave_svm_predict_p_ct.mat');

wave_ann_p = importdata('D:\干旱\data\result8\wave_ann_predict_p.mat');
wave_ann_p_ct = importdata('D:\干旱\data\result8\wave_ann_predict_p_ct.mat');

wave_rf_p = importdata('D:\干旱\data\result8\wave_rf_predict_p.mat');
wave_rf_p_ct = importdata('D:\干旱\data\result8\wave_rf_predict_p_ct.mat');

wave_lstm_p = load('D:\干旱\data\result8\wave_lstm_predict_p.txt');
wave_lstm_p_ct = importdata('D:\干旱\data\result8\wave_lstm_predict_p_ct.txt');
wave_lstm_p = reshape(wave_lstm_p, [m1 m2 m3]);
wave_lstm_p_ct = reshape(wave_lstm_p_ct, [m1 m2 m3]);

wave_conv_lstm_p = load('D:\干旱\data\result8\wave_conv_lstm_predict_p.txt');
wave_conv_lstm_p_ct = importdata('D:\干旱\data\result8\wave_conv_lstm_predict_p_ct.txt');
wave_conv_lstm_p = reshape(wave_conv_lstm_p, [m1 m2 m3]);
wave_conv_lstm_p_ct = reshape(wave_conv_lstm_p_ct, [m1 m2 m3]);

preci = importdata('D:\干旱\data\result8\gpcc_china_6016.mat');%1960-2016,684*2405（65*37）

start_year = 1960;
end_year = 2016;
train_end_year = 1999;
year = (end_year - start_year + 1);

train_end = (train_end_year - start_year + 1) * 12;
all_end = (end_year - start_year + 1) * 12;

% 单个模型在2011-2016年中的效果排序
% [15,16,4,3,20,19,8,7,24,23,12,11,6,18,10,14,5,2,13,17,9,21,22,1]

% 24 个模型
all_predict = cat(4, linear_p, linear_p_ct, wave_linear_p, wave_linear_p_ct, ...
    svm_p, svm_p_ct, wave_svm_p, wave_svm_p_ct, ...
    rf_p, rf_p_ct, wave_rf_p, wave_rf_p_ct, ...
    ann_p, ann_p_ct, wave_ann_p, wave_ann_p_ct, ...
    lstm_p, lstm_p_ct, wave_lstm_p, wave_lstm_p_ct, ...
    conv_lstm_p, conv_lstm_p_ct, wave_conv_lstm_p, wave_conv_lstm_p_ct);

%16 个模型
% all_predict = cat(4, wave_linear_p, wave_linear_p_ct, ...
%     wave_svm_p, wave_svm_p_ct, ...
%     wave_rf_p, wave_rf_p_ct, ...
%     wave_ann_p, wave_ann_p_ct, ...
%     wave_lstm_p, wave_lstm_p_ct, ...
%     wave_conv_lstm_p, wave_conv_lstm_p_ct, ...
% svm_p_ct, lstm_p_ct, rf_p_ct, ann_p_ct);

%12 个模型
% all_predict = cat(4, wave_linear_p, wave_linear_p_ct, ...
%     wave_svm_p, wave_svm_p_ct, ...
%     wave_rf_p, wave_rf_p_ct, ...
%     wave_ann_p, wave_ann_p_ct, ...
%     wave_lstm_p, wave_lstm_p_ct, ...
%     wave_conv_lstm_p, wave_conv_lstm_p_ct);

%8 个模型
% all_predict = cat(4, wave_linear_p, wave_linear_p_ct, ...
%     wave_ann_p, wave_ann_p_ct, ...
%     wave_lstm_p, wave_lstm_p_ct, wave_svm_p, wave_svm_p_ct);

%6 个模型
% all_predict = cat(4, wave_linear_p, wave_linear_p_ct, ...
%     wave_ann_p, wave_ann_p_ct, ...
%     wave_lstm_p, wave_lstm_p_ct);

% % 4 个模型
% all_predict = cat(4, wave_linear_p, wave_linear_p_ct, ...
%     wave_ann_p, wave_ann_p_ct);

%2 个模型
% all_predict = cat(4, wave_ann_p, wave_ann_p_ct);

all_predict( find(all_predict<0) ) = 0; %684*2405*6*24
all_predict( find(isnan(all_predict)) ) = 0;
% save('D:\干旱\data\result8\all_predict_preci.mat', 'all_predict');


out_index = importdata('D:/干旱/data/result8/out_index.mat');
% [m1 m2 m3 m4] = size(all_predict); %month*pts*lead*model
all_predict(:, out_index, :, :) = 0;



[m1 m2 m3 m4] = size(all_predict);
for i=1:m3
    s = all_predict(:,:,i,:);
%     s = squeeze(s);
%     s = permute(s, [2 1 3]);
    s = reshape(s, [m1 m2*m4]);
%     s = reshape(s, [m1 m2*m4]);
%     save( strcat('D:\干旱\data\result8\all_predict_preci_',num2str(i), '.txt'), 's', '-ascii');
%     save( strcat('D:\干旱\data\result8\all_predict_preci_',num2str(i), '_raftery.txt'), 's', '-ascii');
    save( strcat('D:\干旱\data\result8\1960-2010训练24个模型BMA结果\all_predict_preci_',num2str(i), '.txt'), 's', '-ascii');
    disp(i);
end

