function theta = init_parameters(architecture, last_active_is_softmax, varargin )
%基于每一层神经元数量，随机初始化网络中权重参数
% by 郑煜伟 Ewing 2016-04
% architecture: 网络结构；
% theta：权值列向量，[ W1(:); b1(:); W2(:); b2(:); ... ]；

% 没有传入 last_active_is_softmax，默认不是 softmax激活函数
if nargin == 1
    last_active_is_softmax = 0;
end
% 计算参数个数：W个数，b个数；并初始化。
if last_active_is_softmax % softmax那一层不用偏置b
    count_W = architecture * [ architecture(2:end) 0 ]';
    count_B = sum(architecture(2:(end-1)));
    theta = zeros(count_W + count_B, 1);
else
    count_W = architecture * [architecture(2:end) 0]';
    count_B = sum(architecture(2:end));
    theta = zeros(count_W + count_B, 1);
end

% 根据 Hugo Larochelle建议 初始化每层网络的 W
start_index = 1; % 设置每层网络w的下标起点
for layer = 2:length(architecture)
    % 设置每层网络W的下标终点
    end_index = start_index + ...
        architecture(layer)*architecture(layer -1) - 1;
    
    % 权重初始化范围：Hugo Larochelle建议
    r = sqrt(6) / sqrt(architecture(layer) + architecture(layer -1));  
    
    % (layer -1)  -> layer, f( Wx + b )
    theta(start_index:end_index) = rand(architecture(layer) * architecture(layer -1), 1) * 2 * r - r;
    
    % 设置下一层网络W的下标起点（跳过b）
    start_index = end_index + architecture(layer) + 1;
end

end