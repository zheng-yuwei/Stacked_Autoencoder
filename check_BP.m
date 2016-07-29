function [diff, num_gradient, grad] = check_BP(images, labels)
%用于检查sparseAutoencoderEpoch函数所得到的梯度grad是否有效
% by 郑煜伟 Ewing 2016-04
% 我们用数值计算梯度的方法得到梯度numGradient（很慢），
% 与sparseAutoencoderEpoch函数（数学分析方法）得到的梯度（很快）进行比较
% 得到两者梯度向量的欧式距离大小（应该非常之小才对）

image = images(:, 1:1);% 因为计算很慢，所以才抽取一个样本（这个图的theta有308308维！）
label = labels(1, 1);

architecture = [784 196 10]; % AE网络的结构: input_size -> hidden_size -> output_size
last_active_is_softmax = 1;
theta = init_parameters(architecture,...
    last_active_is_softmax); % 依据网络结构初始化网络参数

option.activation  = {'sigmoid', 'softmax'};
option.is_sparse    = 0;
option.sparse_rho   = 0.01;
option.sparse_beta  = 3;
option.is_denoising = 0;
option.decay_lambda = 1;

% 分析方法
[~, grad] = calc_BP_batch(image, label, theta, architecture, option);

% 数值计算方法
num_gradient = compute_numerical_gradient(...
    @(x) calc_BP_batch(image, label, x, architecture, option ), theta);

% 比较梯度的欧式距离
diff = norm(num_gradient - grad) / norm(num_gradient + grad);

end






function num_gradient = compute_numerical_gradient(fun, theta)
%用数值方法计算 函数fun 在 点theta 处的梯度
% fun：输入类theta，输出实值的函数 y = fun(theta)
% theta：参数向量

    % 初始化 num_gradient
    num_gradient = zeros(size(theta));

    % 按微分的原理来计算梯度：变量一个小变化后，函数值得变化程度
    EPSILON    = 1e-4;
    up_theta   = theta;
    down_theta = theta;
    
    wait = waitbar(0, '当前进度');
    for i = 1: length(theta)
        % waitbar( i/length(theta), wait, ['当前进度', num2str(i/length(theta)),'%'] );
        waitbar(i/length(theta), wait);
        
        up_theta(i)    = theta(i) + EPSILON;
        [result_up, ~] = fun(up_theta);
        
        down_theta(i)    = theta(i) - EPSILON;
        [result_down, ~] = fun(down_theta);
        
        num_gradient(i)  = (result_up - result_down) / ( 2 * EPSILON ); % d Vaule / d x
        
        up_theta(i)   = theta(i);
        down_theta(i) = theta(i);
    end
    bar  = findall(get(get(wait, 'children'), 'children'), 'type', 'patch');
    set(bar, 'facecolor', 'g');
    close(wait);
end
