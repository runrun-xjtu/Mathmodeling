
clear all
clc

%%读取数据
data = xlsread('C:\Users\14788\Desktop\Math-Python\输出\2-输出xyz坐标和Z误差-筛选.xlsx');

train_num = floor(0.9 * size(data,1));
randnum=randperm(size(data,1));%产生随机数

data_train=data(randnum(1:train_num),:);%训练样本
data_test=data(randnum(train_num+1:end),:);%测试样本

data_in = data_train(:, 2:4)';
data_out = data_train(:, 6)';
data_intest = data_test(:, 2:4)';
data_outtest = data_test(:, 6)';

%% 第二步 设置训练数据和预测数据

%节点个数
inputnum=2;
hiddennum=8;%隐含层节点数量经验公式p=sqrt(m+n)+a ，故分别取2~13进行试验
outputnum=1;

%% 第三本 训练样本数据归一化
[inputn,inputps]=mapminmax(data_in);%归一化到[-1,1]之间，inputps用来作下一次同样的归一化
[outputn,outputps]=mapminmax(data_out);

%% 第四步 构建BP神经网络
net=newff(inputn,outputn,hiddennum,{'tansig','purelin'},'trainlm');% 建立模型，传递函数使用purelin，采用梯度下降法训练

W1= net. iw{1, 1};%输入层到中间层的权值
B1 = net.b{1};%中间各层神经元阈值

W2 = net.lw{2,1};%中间层到输出层的权值
B2 = net. b{2};%输出层各神经元阈值

%% 第五步 网络参数配置（ 训练次数，学习速率，训练目标最小误差等）
net.trainParam.epochs=1000;         % 训练次数，这里设置为1000次
net.trainParam.lr=0.01;                   % 学习速率，这里设置为0.01
net.trainParam.goal=0.000000001;                    % 训练目标最小误差，这里设置为0.00001

%% 第六步 BP神经网络训练
net=train(net,inputn,outputn);%开始训练，其中inputn,outputn分别为输入输出样本

%% 第七步 测试样本归一化
inputn_test=mapminmax('apply',data_intest,inputps);% 对样本数据进行归一化

%% 第八步 BP神经网络预测
an=sim(net,inputn_test); %用训练好的模型进行仿真

%% 第九步 预测结果反归一化与误差计算     
test_simu=mapminmax('reverse',an,outputps); %把仿真得到的数据还原为原始的数量级
error=test_simu-data_outtest;      %预测值和真实值的误差
error = error';
error_mean = mean(abs(error));
error_std = std(abs(error));
%%第十步 真实值与预测值误差比较
disp(['-----------------------误差计算--------------------------']);
disp(['error_mean = ',num2str(error_mean)]);
disp(['error_std = ',num2str(error_std)]);
