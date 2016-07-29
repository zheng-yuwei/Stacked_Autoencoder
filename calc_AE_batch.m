function [cost,grad] = calc_AE_batch( input, theta, architecture, option_AE, input_corrupted, ~ )
%计算稀疏自编码器的梯度变化和误差
% by 郑煜伟 Ewing 2016-04
% input：       训练样本集，每一列代表一个样本；
% theta：       权值列向量，[ W1(:); b1(:); W2(:); b2(:); ... ]；
% architecture: 网络结构，每层参数组成的行向量
% 结构体 option_AE
% decay_lambda： 权重衰减系数——正则项罚项权重；
% activation：  激活函数类型；

% is_batch_norm： 是否使用 Batch Normalization 来 speed-up学习速度；

% is_sparse：    是否使用 sparse hidden level 的规则；
% sparse_rho：   稀疏性中rho，一般赋值为 0.01；
% sparse_beta：  稀疏性罚项权重；

% input_corrupted： 使用 denoising 规则 则有该参数输入

% 先明确使用AE的规则
% option_AE.is_batch_norm：该规则目前还没加

visible_size = architecture(1);
hidden_size  = architecture(2);
% 先将 theta 转换为 (W1, W2, b1, b2) 的矩阵/向量 形式，以便后续处理（与initializeParameters文件相对应）
W1 = reshape(theta(1 : (hidden_size * visible_size)), ...
    hidden_size, visible_size);
b1 = theta((hidden_size * visible_size + 1) : (hidden_size * visible_size + hidden_size));
W2 = reshape(theta((hidden_size * visible_size + hidden_size + 1) : (2 * hidden_size * visible_size + hidden_size)), ...
    visible_size, hidden_size);
b2 = theta((2 * hidden_size * visible_size + hidden_size + 1) : end);

m = size(input, 2); % 样本数

%% feed forward 阶段
activation_func = str2func(option_AE.activation{:}); % 将 激活函数名 转为 激活函数
% 求隐藏层
if exist('input_corrupted', 'var')
	hidden_V = bsxfun(@plus, W1 * input_corrupted, b1); % 求和 -> V
else
	hidden_V = bsxfun(@plus, W1 * input, b1); % 求和 -> V
end
hidden_X = activation_func(hidden_V); % 激活函数

% 计算隐藏层的稀疏罚项
if option_AE.is_sparse
    rho_hat = sum(hidden_X, 2) / m;
    KL     = get_KL(option_AE.sparse_rho, rho_hat);
    cost_sparse = option_AE.sparse_beta * sum(KL);
else
    cost_sparse = 0;
end

% 求输出层
output_V = bsxfun(@plus, W2 * hidden_X, b2); % 求和 -> V
output_X = activation_func(output_V);   % 激活函数
  
% 求cost function + regularization
cost_error = sum(sum((output_X - input).^2)) / m / 2;
cost_regul = 0.5 * option_AE.decay_lambda * (sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)));  

% 求总的cost
cost = cost_error + cost_regul + cost_sparse;

%% Back Propagation 阶段
activation_func_deriv = str2func([option_AE.activation{:}, '_deriv']);
% 链式法则求导
% dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
dError_dOutputX   = -(input - output_X);
dOutputX_dOutputV = activation_func_deriv(output_V);
dError_dOutputV   = dError_dOutputX .* dOutputX_dOutputV;
% dError/dW2 = dError/dOutputV * dOutputV/dW2
dOutputV_dW2 = hidden_X';
dError_dW2   = dError_dOutputV * dOutputV_dW2;

W2_grad       = dError_dW2 ./ m + option_AE.decay_lambda * W2;
% dError/dHiddenV = (dError/dHiddenX + dSparse/dHiddenX) * dHiddenX/dHiddenV
dError_dHiddenX   = W2' * dError_dOutputV; % = dError/dOutputV * dOutputV/dHiddenX
dHiddenX_dHiddenV = activation_func_deriv(hidden_V);
if option_AE.is_sparse
    dSparse_dHiddenX = option_AE.sparse_beta .* get_KL_deriv(option_AE.sparse_rho, rho_hat);
    dError_dHiddenV  = (dError_dHiddenX + repmat(dSparse_dHiddenX, 1, m)) .* dHiddenX_dHiddenV;
else
    dError_dHiddenV  = dError_dHiddenX .* dHiddenX_dHiddenV;
end
% dError/dW1 = dError/dHiddenV * dHiddenV/dW1
dHiddenV_dW1 = input';
dError_dW1   = dError_dHiddenV * dHiddenV_dW1;

W1_grad       = dError_dW1 ./ m + option_AE.decay_lambda * W1;


% 用于解释梯度消失得厉害！！！
% disp('梯度消失');
% disp(['W2梯度绝对值均值：', num2str(mean(mean(abs(W2_grad)))), ...
%     ' -> ','W1梯度绝对值均值：', num2str(mean(mean(abs(W1_grad))))]);
% disp(['W2梯度最大值：', num2str(max(mean(W2_grad))), ...
%     ' -> ','W1梯度最大值：', num2str(max(mean(W1_grad)))]);


% 求偏置的导数
dError_db2 = sum(dError_dOutputV, 2);
b2_grad     = dError_db2 ./ m;
dError_db1 = sum(dError_dHiddenV, 2);  
b1_grad     = dError_db1 ./ m;

grad = [W1_grad(:); b1_grad(:); W2_grad(:); b2_grad(:)];

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

%% KL散度函数及导数
function KL = get_KL(sparse_rho,rho_hat)
%KL-散度函数
    EPSILON = 1e-8; %防止除0
    KL = sparse_rho .* log( sparse_rho ./ (rho_hat + EPSILON) ) + ...
        ( 1 - sparse_rho ) .* log( (1 - sparse_rho) ./ (1 - rho_hat + EPSILON) );  
end

function KL_deriv = get_KL_deriv(sparse_rho,rho_hat)
%KL-散度函数的导数
    EPSILON = 1e-8; %防止除0
    KL_deriv = ( -sparse_rho ) ./ ( rho_hat + EPSILON ) + ...
        ( 1 - sparse_rho ) ./ ( 1 - rho_hat + EPSILON );  
end