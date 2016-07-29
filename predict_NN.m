function predict_labels = predict_NN(input, architecture, theta, option)
%计算网络前向阶段，以实现预测
% by 郑煜伟 Ewing 2016-04

start_index = 1; % 存储变量的下标起点
for i = 1:(length(architecture) - 1)
    visible_size = architecture(i);
    hidden_size  = architecture(i + 1);
    
    %% 先将 theta 转换为 (W, b) 的矩阵/向量 形式，以便后续处理（与init_parameters文件相对应）
    end_index = hidden_size * visible_size + start_index - 1; % 存储变量的下标终点
    W = reshape(theta(start_index : end_index), hidden_size, visible_size);
    
    if strcmp(option.activation{i}, 'softmax') % softmax不需要偏置b
        start_index = end_index + 1; % 存储变量的下标起点
    else
        start_index = end_index + 1; % 存储变量的下标起点
        end_index = hidden_size + start_index - 1; % 存储变量的下标终点
        b = theta(start_index : end_index);
        start_index = end_index + 1;
    end
    
    %% feed forward 阶段
    activation_func = str2func(option.activation{i}); % 将 激活函数名 转为 激活函数
    % 求隐藏层
    if strcmp(option.activation{i}, 'softmax') % softmax不需要偏置b
        hidden_V = W * input; % 求和 -> 诱导局部域V
    else
        hidden_V = bsxfun(@plus, W * input, b); % 求和 -> 诱导局部域V
    end
    hidden_X = activation_func(hidden_V); % 激活函数
    
    clear input
    input = hidden_X;
end

predict_labels = input;

end


%% 激活函数
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));  
end
% tanh有自带函数
function x = ReLU(x)
    x(x < 0) = 0;
end
function x = weakly_ReLU(x)
    x(x < 0) = x(x < 0) * 0.2;
end
function soft = softmax(x)
    soft = exp(x);
    soft = bsxfun(@rdivide, soft, sum(soft, 1));
end