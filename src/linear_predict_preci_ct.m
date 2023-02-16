%对降雨进行预测，linear_ct
%linear, 不分每月进行预测------------
clear all;
clc;
%1960-2016
preci = importdata('D:\干旱\data\result8\gpcc_china_6016.mat');%1960-2016,684*2405（65*37）
start_year = 1960;
end_year = 2016;
train_end_year = 2010;
year = (end_year - start_year + 1);

train_end = (train_end_year - start_year + 1) * 12;
all_end = (end_year - start_year + 1) * 12;

train_label_data = preci(1:train_end,:);
[m n] = size(train_label_data);

tem = importdata('D:\干旱\data\result8\cruts_tem_6016.mat');
tem_min = importdata('D:\干旱\data\result8\cruts_tem_min_6016.mat');
tem_max = importdata('D:\干旱\data\result8\cruts_tem_max_6016.mat');
[m1 m2 m3] = size(tem);
tem = reshape(tem, [m1 m2*m3]);
tem_min = reshape(tem_min, [m1 m2*m3]);
tem_max = reshape(tem_max, [m1 m2*m3]);

%读取气候指标
all_climate_index = importdata('D:\干旱\data\result8\all_climate_index.mat');
train_data_ci = all_climate_index(1:train_end,:);

train_label_data = preci(1:train_end,:);
[m n] = size(train_label_data);
%每个位置建立一个SVR模型
%用前5个数据做输入，最后一个作为输出，如果预测提前3个月的，则循环来做
lead = 6;

[m2 n2] = size(preci);
linear_ct_predict = NaN(m2,n2,lead);%行*列*lead  提前1-8个月的预测值
for i=1:n
    for j=1:lead
        x1 = [NaN(j,1); preci(1:train_end-j,i)];%提前1个月降雨
        x2 = [NaN(j,1); tem(1:train_end-j,i)];%提前2个月降雨
        x3 = [NaN(j,1); tem_min(1:train_end-j,i)];
        x4 = [NaN(j,1); tem_max(1:train_end-j,i)];
%         x5 = [NaN(j,size(train_data_ci,2)); train_data_ci(1:end-j,:)];
        train_sub = [x1 x2 x3 x4];
        label_sub = train_label_data(1:end,i);
        
        if isempty( find(~isnan(label_sub)) )
            continue;
        end
        flag = 1;
        for k=1:size(train_sub, 2)
            if isempty( find(~isnan(train_sub(:,k))) )
                flag = 0;
            end
        end
        if flag==0
            continue;
        end
        
        %train_sub,n*p的矩阵，n为观测值个数，p为变量个数，label_sub为n*1矩阵
%         Mdl = fitrsvm(train_sub,label_sub,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
        %     Mdl = fitrsvm(X,Y,'KernelFunction','rbf','KernelScale','auto','Standardize',true,'OptimizeHyperparameters','auto',...
        %     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        %     'expected-improvement-plus'))
%         [b,bint,r,rint,stats] = regress(label_sub,[ones(size(train_sub,1), 1) train_sub]);%线性模型
        
        %所有点所有月份提前j个月的预测，1->1+j, end->end+j，即第1个月预测的是第j个月的
%         x_pre = [all_climate_index(1:end, :)];
        x1 = [NaN(j,1); preci(1:end-j,i)];%提前1个月降雨
        x2 = [NaN(j,1); tem(1:end-j,i)];%提前2个月降雨
        x3 = [NaN(j,1); tem_min(1:end-j,i)];
        x4 = [NaN(j,1); tem_max(1:end-j,i)];
        
        x_pre = [x1 x2 x3 x4];
        
        train_sub_ci = [NaN(j,size(train_data_ci,2)); train_data_ci(1:end-j,:)];
        x_pre_ci = [NaN(j,size(train_data_ci,2)); all_climate_index(1:end-j, :)];
        
        train_sub = [train_sub train_sub_ci];
        x_pre_new = [x_pre x_pre_ci];
        x_pre = x_pre_new;
        
        %train_sub,n*p的矩阵，n为观测值个数，p为变量个数，label_sub为n*1矩阵
        [b,bint,r,rint,stats] = regress(label_sub,[ones(size(train_sub,1), 1) train_sub]);%线性模型
        pre_data_2 = [ones(size(x_pre,1), 1) x_pre];
        y_pre_2 = pre_data_2 * b;
        linear_ct_predict(:, i, j) = y_pre_2; %第i个点提前第j个月的预测
    end
    disp(i);
end

save('D:\干旱\data\result8\linear_predict_p_ct.mat','linear_ct_predict');
disp('finished');
