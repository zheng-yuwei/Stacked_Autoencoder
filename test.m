%测试用的文件
% by 郑煜伟 Ewing 2016-04

%% 验证AE梯度计算的正确性
% diff = check_AE(images); % 已经验证，除非有时间，别运行！慢！天的那种
% disp(diff); % diff应该很小

%% 测试 sparse DAE：训练一层 sparse DAE，将重构数据与原数据进行对比 - DAE通过
clc;clear
% 用到 loadMNISTImages，getAEOption，initializeParameters，trainAE函数
[input, labels] = load_MNIST_data( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'min_max_scaler', 1 );
architecture = [784 196 784];
% 设置 AE的预选参数 及 BP的预选参数
preOption_SAE.option_AE.is_sparse    = 1;
preOption_SAE.option_AE.is_denoising = 1;
preOption_SAE.option_AE.activation  = {'ReLU'};
% 得到SAE的预选参数
option_SAE = get_SAE_option(preOption_SAE);
option_AE  = option_SAE.option_AE;

count_AE = 1;

theta = init_parameters(architecture);
[opt_theta, cost] = trainAE(input, theta, architecture, count_AE, option_AE);

% 将训练好的AE所重构出来的图片输出，与原始图片进行对比
option_AE.activation = {'ReLU'; 'ReLU'};
predict = predict_NN( input, architecture, opt_theta, option_AE );

images_predict = reshape(predict, sqrt(size(predict, 1)), sqrt(size(predict, 1)), size(predict, 2));
% 灰度图
figure('NumberTitle', 'off', 'Name', 'MNIST手写字体图片(重构）');
show_images_num = 200;
penal           = show_images_num * 2 / 3;
pic_mat_col     = ceil(1.5 * sqrt(penal));
pic_mat_row     = ceil(show_images_num / pic_mat_col);
for i = 1:show_images_num
    subplot(pic_mat_row, pic_mat_col, i, 'align' );
    imshow(images_predict(:, :, i));
end
% 热量图 jet
figure('NumberTitle', 'off', 'Name', 'MNIST手写字体图片(重构）-热量图');
for i = 1:show_images_num
    subplot(pic_mat_row, pic_mat_col, i, 'align');
    imagesc(images_predict(:, :, i));
    axis off;
end



