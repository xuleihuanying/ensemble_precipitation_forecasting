% linear Ԥ�⽵��
% linear, ����ÿ�½���Ԥ��------------
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
%ÿ��λ�ý���һ��SVRģ��
%��ǰ5�����������룬���һ����Ϊ��������Ԥ����ǰ3���µģ���ѭ������
lead = 6;

[m2 n2] = size(preci);
linear_predict = NaN(m2,n2,lead);%��*��*lead  ��ǰ1-8���µ�Ԥ��ֵ

for i=1:n
    for j=1:lead
        x1 = [NaN(j,1); preci(1:train_end-j,i)];%��ǰ1���½���
        x2 = [NaN(j+1,1); preci(1:train_end-j-1,i)];%��ǰ2���½���
        x3 = [NaN(j+2,1); preci(1:train_end-j-2,i)];
        x4 = [NaN(j+3,1); preci(1:train_end-j-3,i)];
        x5 = [NaN(j+4,1); preci(1:train_end-j-4,i)];
        x6 = [NaN(j+5,1); preci(1:train_end-j-5,i)];
        
        train_sub = [x1 x2 x3 x4 x5 x6];%���ϳ�ʼ����
        label_sub = train_label_data(1:end,i);
        
        if isempty( find(~isnan(label_sub)) )
            continue;
        end
        
%         train_sub(find(isnan(train_sub)) ) = 0;
        
        %���е������·���ǰj���µ�Ԥ�⣬1->1+j, end->end+j������1����Ԥ����ǵ�j���µ�
%         x_pre = [all_climate_index(1:end, :)];
        x1 = [NaN(j,1); preci(1:end-j,i)];%��ǰ1���½���
        x2 = [NaN(j+1,1); preci(1:end-j-1,i)];%��ǰ2���½���
        x3 = [NaN(j+2,1); preci(1:end-j-2,i)];
        x4 = [NaN(j+3,1); preci(1:end-j-3,i)];
        x5 = [NaN(j+4,1); preci(1:end-j-4,i)];
        x6 = [NaN(j+5,1); preci(1:end-j-5,i)];
        x_pre = [x1 x2 x3 x4 x5 x6];
        
        %train_sub,n*p�ľ���nΪ�۲�ֵ������pΪ����������label_subΪn*1����
        [b,bint,r,rint,stats] = regress(label_sub,[ones(size(train_sub,1), 1) train_sub]);%����ģ��
        pre_data_2 = [ones(size(x_pre,1), 1) x_pre];
        y_pre_2 = pre_data_2 * b;
        linear_predict(:, i, j) = y_pre_2; %��i������ǰ��j���µ�Ԥ��
    end
    disp(i);
end

save('D:\�ɺ�\data\result8\linear_predict_p.mat','linear_predict');
disp('finished');