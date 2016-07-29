function [cost,grad] = calc_BP_batch(input, output, theta, architecture, option)
%计算 BPNN 的梯度变化和误差
% by 郑煜伟 Ewing 2016-04
% input：       训练样本集，每一列代表一个样本；
% theta：       权值列向量，[ W1(:); b1(:); W2(:); b2(:); ... ]；
% architecture: 网络结构，每层参数组成的行向量
% 结构体 option
% decay_lambda：      权重衰减系数——正则项罚项权重；
% activation：  激活函数类型；

% is_batch_norm：   是否使用 Batch Normalization 来 speed-up学习速度；

% 先明确使用BP的规则
% option.is_batch_norm：该规则目前还没加


m                = size(input, 2); % 样本数
layers           = length(architecture); % 网络层数
% 初始化一些参数
layer_hidden_V     = cell(1, layers - 1); % 用于盛装每一层神经网络的诱导局部域数据
layer_hidden_X     = cell(1, layers);     % 用于盛装每一层神经网络的输出/输入数据
layer_hidden_X{1}  = input;
cost_regul         = 0; % 正则项的罚函数
cost_error         = 0; % cost function
grad               = zeros(size(theta));
%% feed-forward阶段
start_index = 1; % 存储变量的下标起点
for i = 1:(layers - 1)
    visible_size = architecture(i);
    hidden_size  = architecture(i + 1);
    
    activation_func = str2func(option.activation{i}); % 将 激活函数名 转为 激活函数
    
    % 先将 theta 转换为 (W, b) 的矩阵/向量 形式，以便后续处理（与initializeParameters文件相对应）
    end_index  = hidden_size * visible_size + start_index - 1; % 存储变量的下标终点
    W          = reshape(theta(start_index : end_index), hidden_size, visible_size);
    
    if strcmp(option.activation{i}, 'softmax') % softmax那一层不用偏置b
        start_index = end_index + 1; % 存储变量的下标起点
        
        hidden_V = W * input;% 求和 -> 得到诱导局部域 V
    else
        start_index = end_index + 1; % 存储变量的下标起点
        end_index   = hidden_size + start_index - 1; % 存储变量的下标终点
        b           = theta(start_index : end_index);
        start_index = end_index + 1;
        
        hidden_V = bsxfun(@plus, W * input, b); % 求和 -> 得到诱导局部域 V
    end
    hidden_X = activation_func(hidden_V); % 激活函数
    % 计算正则项的罚函数
    cost_regul = cost_regul + 0.5 * option.decay_lambda * sum(sum(W .^ 2));
    
    clear input
    input = hidden_X;
    
    layer_hidden_V{i}     = hidden_V; % 用于盛装每一层神经网络的诱导局部域数据
    layer_hidden_X{i + 1} = input;   % 用于盛装每一层神经网络的输出/输入数据
end
% 求cost function + regularization
if strcmp(option.activation{end}, 'softmax') % 标签类cost
    % softmax的cost，但我并没有求对数，并且加了1. 用于模仿准确率
    index_row = output';
    index_col = 1:m;
    index    = (index_col - 1) .* architecture(end) + index_row;
%     cost_error = sum(1 - layer_hidden_X{layers}(index)) / m;
    cost_error = - sum(log(layer_hidden_X{layers}(index))) / m; 
else % 实值类cost
    cost_error = sum(sum((output - layer_hidden_X{layers}).^2)) ./ 2 / m;
end

cost = cost_error + cost_regul;

%% Back Propagation 阶段：链式法则求导
% 求最后一层
activation_func_deriv = str2func([option.activation{layers-1}, '_deriv']);
if strcmp(option.activation{layers-1}, 'softmax') % softmax那一层求导需要额外labels信息
    dError_dOutputV   = activation_func_deriv(layer_hidden_V{layers - 1}, output);
else
    % dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
    dError_dOutputX   = -(output - layer_hidden_X{layers});
    dOutputX_dOutputV = activation_func_deriv(layer_hidden_V{layers - 1});
    dError_dOutputV   = dError_dOutputX .* dOutputX_dOutputV;
end


% dError/dW = dError/dOutputV * dOutputV/dW
dOutputV_dW = layer_hidden_X{layers - 1}';
dError_dW   = dError_dOutputV * dOutputV_dW;

if strcmp(option.activation{layers-1}, 'softmax') % softmax那一层不用偏置b
    end_index   = length(theta); % 存储变量的下标终点
    start_index = end_index + 1; % 存储变量的下标起点
else
    % 更新梯度 b
    end_index   = length(theta); % 存储变量的下标终点
    start_index = end_index - architecture(end)  + 1; % 存储变量的下标起点
    dError_db  = sum(dError_dOutputV, 2);
    grad(start_index:end_index) = dError_db ./ m;
end
% 更新梯度 W
end_index   = start_index - 1; % 存储变量的下标终点
start_index = end_index - architecture(end - 1) * architecture(end)  + 1; % 存储变量的下标起点
W          = reshape(theta(start_index:end_index), architecture(end), architecture(end - 1 ));
W_grad      = dError_dW ./ m + option.decay_lambda * W;
grad(start_index:end_index) = W_grad(:);

% 误差回传 error back-propagation
for i = (layers - 2):-1:1
    activation_func_deriv = str2func([option.activation{i}, '_deriv']);
    % dError/dHiddenV = dError/dHiddenX * dHiddenX/dHiddenV
    dError_dHiddenX   = W' * dError_dOutputV; % = dError/dOutputV * dOutputV/dHiddenX
    dHiddenX_dHiddenV = activation_func_deriv(layer_hidden_V{i});
    dError_dHiddenV   = dError_dHiddenX .* dHiddenX_dHiddenV;
    % dError/dW1 = dError/dHiddenV * dHiddenV/dW1
    dHiddenV_dW = layer_hidden_X{i}';
    dError_dW   = dError_dHiddenV * dHiddenV_dW;
    
    dError_db = sum(dError_dHiddenV, 2);
    % 更新梯度 b
    end_index   = start_index - 1; % 存储变量的下标终点
    start_index = end_index - architecture(i + 1)  + 1; % 存储变量的下标起点
    % b          = theta( startIndex : endIndex );
    grad(start_index:end_index) = dError_db ./ m;
    
    % 更新梯度 W
    end_index   = start_index - 1; % 存储变量的下标终点
    start_index = end_index - architecture( i ) * architecture( i + 1 )  + 1; % 存储变量的下标起点
    W          = reshape( theta(start_index:end_index), architecture( i + 1 ), architecture( i ) );
    W_grad      = dError_dW ./ m + option.decay_lambda * W;
    grad(start_index:end_index) = W_grad(:);
    
    dError_dOutputV = dError_dHiddenV;
end

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
%% 激活函数导数
function sigm_deriv = sigmoid_deriv(x)
    sigm_deriv = sigmoid(x).*(1-sigmoid(x));  
end
function tan_deriv = tanh_deriv(x)
    tan_deriv = 1 ./ cosh(x).^2; % tanh的导数
end
function x = ReLU_deriv(x)
    x(x < 0) = 0;
    x(x > 0) = 1;
end
function x = weakly_ReLU_deriv(x)
    x(x < 0) = 0.2;
    x(x > 0) = 1;
end
function soft_deriv = softmax_deriv( x, labels )
    index_row = labels';
    index_col = 1:length(index_row);
    index    = (index_col - 1) .* max(labels) + index_row;
    
%     softDeriv = softmax(x);
%     active   = zeros( size(x) );
%     active(index) = 1;
%     softDeriv = bsxfun( @times, softDeriv - active, softDeriv(index) );

    soft_deriv = softmax(x);
    soft_deriv(index) = soft_deriv(index) - 1;  % 这个是使用原始cost function的导数
end







