%�Խ������Ԥ�⣬linear_ct
%linear, ����ÿ�½���Ԥ��------------
clear all;
clc;
%1960-2016
preci = importdata('D:\�ɺ�\data\result8\gpcc_china_6016.mat');%1960-2016,684*2405��65*37��
start_year = 1960;
end_year = 2016;
train_end_year = 2010;
year = (end_year - start_year + 1);

train_end = (train_end_year - start_year + 1) * 12;
all_end = (end_year - start_year + 1) * 12;

train_label_data = preci(1:train_end,:);
[m n] = size(train_label_data);

tem = importdata('D:\�ɺ�\data\result8\cruts_tem_6016.mat');
tem_min = importdata('D:\�ɺ�\data\result8\cruts_tem_min_6016.mat');
tem_max = importdata('D:\�ɺ�\data\result8\cruts_tem_max_6016.mat');
[m1 m2 m3] = size(tem);
tem = reshape(tem, [m1 m2*m3]);
tem_min = reshape(tem_min, [m1 m2*m3]);
tem_max = reshape(tem_max, [m1 m2*m3]);

%��ȡ����ָ��
all_climate_index = importdata('D:\�ɺ�\data\result8\all_climate_index.mat');
train_data_ci = all_climate_index(1:train_end,:);

train_label_data = preci(1:train_end,:);
[m n] = size(train_label_data);
%ÿ��λ�ý���һ��SVRģ��
%��ǰ5�����������룬���һ����Ϊ��������Ԥ����ǰ3���µģ���ѭ������
lead = 6;

[m2 n2] = size(preci);
linear_ct_predict = NaN(m2,n2,lead);%��*��*lead  ��ǰ1-8���µ�Ԥ��ֵ
for i=1:n
    for j=1:lead
        x1 = [NaN(j,1); preci(1:train_end-j,i)];%��ǰ1���½���
        x2 = [NaN(j,1); tem(1:train_end-j,i)];%��ǰ2���½���
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
        
        %train_sub,n*p�ľ���nΪ�۲�ֵ������pΪ����������label_subΪn*1����
%         Mdl = fitrsvm(train_sub,label_sub,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
        %     Mdl = fitrsvm(X,Y,'KernelFunction','rbf','KernelScale','auto','Standardize',true,'OptimizeHyperparameters','auto',...
        %     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        %     'expected-improvement-plus'))
%         [b,bint,r,rint,stats] = regress(label_sub,[ones(size(train_sub,1), 1) train_sub]);%����ģ��
        
        %���е������·���ǰj���µ�Ԥ�⣬1->1+j, end->end+j������1����Ԥ����ǵ�j���µ�
%         x_pre = [all_climate_index(1:end, :)];
        x1 = [NaN(j,1); preci(1:end-j,i)];%��ǰ1���½���
        x2 = [NaN(j,1); tem(1:end-j,i)];%��ǰ2���½���
        x3 = [NaN(j,1); tem_min(1:end-j,i)];
        x4 = [NaN(j,1); tem_max(1:end-j,i)];
        
        x_pre = [x1 x2 x3 x4];
        
        train_sub_ci = [NaN(j,size(train_data_ci,2)); train_data_ci(1:end-j,:)];
        x_pre_ci = [NaN(j,size(train_data_ci,2)); all_climate_index(1:end-j, :)];
        
        train_sub = [train_sub train_sub_ci];
        x_pre_new = [x_pre x_pre_ci];
        x_pre = x_pre_new;
        
        %train_sub,n*p�ľ���nΪ�۲�ֵ������pΪ����������label_subΪn*1����
        [b,bint,r,rint,stats] = regress(label_sub,[ones(size(train_sub,1), 1) train_sub]);%����ģ��
        pre_data_2 = [ones(size(x_pre,1), 1) x_pre];
        y_pre_2 = pre_data_2 * b;
        linear_ct_predict(:, i, j) = y_pre_2; %��i������ǰ��j���µ�Ԥ��
    end
    disp(i);
end

save('D:\�ɺ�\data\result8\linear_predict_p_ct.mat','linear_ct_predict');
disp('finished');
