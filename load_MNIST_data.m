function [images, labels] = load_MNIST_data(images_file, labels_file,...
    preprocess, is_show_images, varargin)
%加载MNIST数据集：images和labels
% by 郑煜伟 Ewing 2016-04

if exist('is_show_images', 'var')
    images = load_MNIST_images(images_file, preprocess, is_show_images);
else
    images = load_MNIST_images(images_file);
end
labels = load_MNIST_labels(labels_file);

end

function images = load_MNIST_images(file_name, preprocess, is_show_images, varargin)
%返回一个  #像素点数 * #样本数 的矩阵

    %% 读取 raw MNIST images
    fp = fopen(file_name, 'rb');
    assert(fp ~= -1, ['Could not open ', file_name, ' ']);  % 打不开则报错

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', file_name, ' ']); % 规定的 magic number，用于check文件是否正确

    num_images = fread(fp, 1, 'int32', 0, 'ieee-be'); % 连续读出三个关于文件数据属性的数
    num_rows   = fread(fp, 1, 'int32', 0, 'ieee-be');
    num_cols   = fread(fp, 1, 'int32', 0, 'ieee-be');

    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, num_cols, num_rows, num_images); % 文件数据是按行排列的，而matlab是按列排列的。
    images = permute(images, [ 2 1 3 ]);

    fclose(fp);
    %% 显示200张images
    if exist('is_show_images', 'var') &&  is_show_images == 1
        figure('NumberTitle', 'off', 'Name', 'MNIST手写字体图片');
        show_images_num = 200;
        penal           = show_images_num * 2 / 3;
        pic_Mat_Col     = ceil(1.5 * sqrt(penal));
        pic_Mat_Row     = ceil(show_images_num / pic_Mat_Col);
        for i = 1:show_images_num
            subplot(pic_Mat_Row, pic_Mat_Col, i, 'align');
            imshow(images(:, :, i));
        end
    end

    %% 对 images 进行处理
    % 转化为 #像素点数 * #样本数 矩阵
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    
    if strcmp(preprocess, 'min_max_scaler')
        % 归一化到 [0,1]
        images = double(images) / 255; % 激活函数值域非负
    elseif strcmp(preprocess, 'zScore')
        % 标准化处理
        images = zScore( images );% 激活函数值域可正可负
    elseif strcmp(preprocess, 'whitening')
        % 白化
        images = whitening( images ); % 激活函数值域可正可负
    end
end

function data = zScore(data)
%对数据进行标准化处理（样本按列排列）
% 去均值，然后方差缩放
    epsilon = 1e-8; % 防止除0
    data = bsxfun(@minus, data, mean(data, 1)); % 去均值（这里类似去除图片亮度）
    data = bsxfun(@rdivide, data, sqrt(mean(data .^ 2, 1)) + epsilon); % 去方差
end
function data = whitening(data)
%对数据进行白化处理（样本按列排列）
% 去均值，然后去相关性
    data = bsxfun(@minus, data, mean(data, 1)); % 去均值
    [u, s, ~] = svd(data * data' / size(data, 2)) ; % 求协方差矩阵的svd分解
    data = sqrt(s) \ u' * data; % 白化（去相关性，协方差为1）
end

function labels = load_MNIST_labels( file_name )
%返回一个 #标签数 * #1 的列向量

    %% 读取 raw MNIST labels
    fp = fopen(file_name, 'rb');
    assert(fp ~= -1, ['Could not open ', file_name, ' ' ]);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', file_name, ' ']);

    num_labels = fread(fp, 1, 'int32', 0, 'ieee-be');
    labels = fread(fp, inf, 'unsigned char');

    assert(size(labels, 1) == num_labels, 'Mismatch in label count');
    fclose(fp);

    labels(labels == 0) = 10;

    % 下面本想化成矩阵形式的，后面用softmax就没化了
    % index_row     = labels';
    % index_col     = 1:num_labels;
    % index         = (index_col - 1) .* 10 + index_row;
    % labels        = zeros(10, num_labels);
    % labels(index) = 1;
end



