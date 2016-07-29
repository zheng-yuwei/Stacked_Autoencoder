function display_network( weight, figure_name, ~ )
%利用网络权重(hidden_size*input_size)展示网络所抽取的特征图
% 假设每个 hidden level 1 的 neuron 表示所抽取的一种 feature
% 则连接到 neuron A 的权重向量，代表 input vector 中每一位在 feature A 的重要程度
% 根据权重向量（重要程度），即可构造出 input 的 feature
% by 郑煜伟 Ewing 2016-04

% 对 每个input位权重 实施归一化
weight_min = min(weight, [], 2);
weight     = bsxfun(@minus, weight, weight_min);
weight_max = max( weight, [], 2 );
weight     = bsxfun(@rdivide, weight, weight_max);

feature_num   = size(weight, 1); % feature数量，也是图片数量
penal         = feature_num * 2 / 3;
pic_mat_col   = ceil(1.5 * sqrt(penal));
pic_mat_row   = ceil(feature_num / pic_mat_col);

images = reshape(weight', sqrt(size(weight, 2)), sqrt(size(weight, 2)), feature_num); % 图片
% 展示特征
% 灰度图
if exist('figure_name', 'var')
    figure('NumberTitle', 'off', 'Name', figure_name);
else
    figure('NumberTitle', 'off', 'Name', 'MNIST手写字体特征图');
end
for i = 1:feature_num
    subplot( pic_mat_row, pic_mat_col, i, 'align' );
    imshow( images(:, :, i) );
%     imagesc( images(:, :, i) );
%     axis off;
end

end