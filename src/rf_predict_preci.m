% random forest 预测降雨
% random forest, 不分每月进行预测------------
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
%每个位置建立一个SVR模型
%用前5个数据做输入，最后一个作为输出，如果预测提前3个月的，则循环来做
lead = 6;

[m2 n2] = size(preci);
rf_predict = NaN(m2,n2,lead);%行*列*lead  提前1-8个月的预测值

tree_num = 20; % tree number
tree_v_sct = 4; % selected number of predictors per run
for i=1:n
    for j=1:lead
        x1 = [NaN(j,1); preci(1:train_end-j,i)];%提前1个月降雨
        x2 = [NaN(j+1,1); preci(1:train_end-j-1,i)];%提前2个月降雨
        x3 = [NaN(j+2,1); preci(1:train_end-j-2,i)];
        x4 = [NaN(j+3,1); preci(1:train_end-j-3,i)];
        x5 = [NaN(j+4,1); preci(1:train_end-j-4,i)];
        x6 = [NaN(j+5,1); preci(1:train_end-j-5,i)];
        
        train_sub = [x1 x2 x3 x4 x5 x6];%加上初始条件
        label_sub = train_label_data(1:end,i);
        
        if isempty( find(~isnan(label_sub)) )
            continue;
        end
        
        %train_sub,n*p的矩阵，n为观测值个数，p为变量个数，label_sub为n*1矩阵
        Mdl = TreeBagger(tree_num, train_sub, label_sub,'Method','regression','Surrogate','on',...
    'OOBPredictorImportance','on','NumPredictorsToSample','all', 'MinLeafSize', 5); 
%         Mdl = fitrsvm(train_sub,label_sub,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
%         %     Mdl = fitrsvm(X,Y,'KernelFunction','rbf','KernelScale','auto','Standardize',true,'OptimizeHyperparameters','auto',...
%         %     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%         %     'expected-improvement-plus'))
% %         [b,bint,r,rint,stats] = regress(label_sub,[ones(size(train_sub,1), 1) train_sub]);%线性模型
%         svm_para_save(i, j, 1) = Mdl.BoxConstraints(1);
%         svm_para_save(i, j, 2) = Mdl.Epsilon;
%         svm_para_save(i, j, 3) = Mdl.KernelParameters.Scale;
        %所有点所有月份提前j个月的预测，1->1+j, end->end+j，即第1个月预测的是第j个月的
%         x_pre = [all_climate_index(1:end, :)];
        x1 = [NaN(j,1); preci(1:end-j,i)];%提前1个月降雨
        x2 = [NaN(j+1,1); preci(1:end-j-1,i)];%提前2个月降雨
        x3 = [NaN(j+2,1); preci(1:end-j-2,i)];
        x4 = [NaN(j+3,1); preci(1:end-j-3,i)];
        x5 = [NaN(j+4,1); preci(1:end-j-4,i)];
        x6 = [NaN(j+5,1); preci(1:end-j-5,i)];
        x_pre = [x1 x2 x3 x4 x5 x6];
        
%         for month = 1:12
%             train_sub_new = train_sub(month:12:end, :);
%             label_sub_new = label_sub(month:12:end, :);
%             Mdl = fitrsvm(train_sub_new, label_sub_new,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
% %             [b,bint,r,rint,stats] = regress(label_sub_new,[ones(size(train_sub_new,1), 1) train_sub_new]);%线性模型
%             x_pre_new = x_pre(month:12:end, :);
%             ynew = predict(Mdl,x_pre_new); %第i个点提前第j个月的预测
%             rf_predict(month:12:end, i, j) = ynew; %第i个点提前第j个月的预测
%         end
        
%         pre_data_2 = [ones(size(x_pre,1), 1) x_pre];
%         y_pre_2 = [];
%         for h=1:size(pre_data_2,1)
%             y_pre_2(h,:) = pre_data_2(h,:) * b;
%         end
        
        ynew = predict(Mdl,x_pre); %第i个点提前第j个月的预测
        rf_predict(:, i, j) = ynew; %第i个点提前第j个月的预测
    end
    disp(i);
end

save('D:\干旱\data\result8\rf_predict_p.mat','rf_predict');
disp('finished');