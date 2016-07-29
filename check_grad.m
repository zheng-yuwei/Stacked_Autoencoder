clc, clear;
% by 郑煜伟 Ewing 2016-04

% 加载数据
[ images_train, labels_train ] = load_MNIST_data( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'min_max_scaler', 0 );

% 检测 计算AE梯度的准确性
[diff, num_gradient, grad] = check_AE(images_train);
fprintf(['AE中计算梯度的分析方法与数值方法的差异性：'...
    num2str(mean(abs(num_gradient - grad)))...
    ' 及 ' num2str(diff) '\n']);

% 检测 计算BP梯度的准确性
[diff, num_gradient, grad] = check_BP(images_train, labels_train);
fprintf(['AE中计算梯度的分析方法与数值方法的差异性：'...
    num2str(mean(abs(num_gradient - grad)))...
    ' 及 ' num2str(diff) '\n']);