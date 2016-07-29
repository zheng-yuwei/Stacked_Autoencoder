function option_BPNN = get_BPNN_option(preOption_BPNN)
%设置BP的参数
% by 郑煜伟 Ewing 2016-04
% 输入 BP网络的选项 preOption_BPNN
% 返回：
% AE网络的选项： option_BPNN
% decay_lambda：   权重衰减系数——正则项罚项权重；
% activation：     激活函数类型；

% is_batch_norm：  是否使用 Batch Normalization 来 speed-up学习速度；

% is_denoising：   是否使用 denoising 规则
% noise_layer：    AE中添加噪声的层：'first_layer' or 'all_layers'
% noise_rate：     每一位添加噪声的概率
% noise_mode：     添加噪声的模式：'On_Off' or 'Guass'
% noise_mean：     高斯模式：均值
% noise_sigma：    高斯模式：标准差

if isfield(preOption_BPNN, 'decay_lambda')
	option_BPNN.decay_lambda = preOption_BPNN.decay_lambda;
else
	option_BPNN.decay_lambda = 0.001;
end

if isfield(preOption_BPNN, 'activation')
	option_BPNN.activation = preOption_BPNN.activation;
else
	error('激活函数列表必须由你自己来定！');
end

% batch normalization
if isfield(preOption_BPNN, 'is_batch_norm')
	option_BPNN.is_batch_norm = preOption_BPNN.is_batch_norm;
else
	option_BPNN.is_batch_norm = 0;
end

% de-noising
if isfield(option_BPNN, 'is_denoising')
    option_BPNN.is_denoising = option_BPNN.is_denoising;
    if option_BPNN.is_denoising
        % denoising每一层 或 只第一个输入层
        if isfield(option_BPNN, 'noise_layer')
            option_BPNN.noise_layer = option_BPNN.noise_layer;
        else
            option_BPNN.noise_layer = 'first_layer';
        end
        % 噪声概率
        if isfield(option_BPNN, 'noise_rate')
            option_BPNN.noise_rate = option_BPNN.noise_rate;
        else
            option_BPNN.noise_rate = 0.1;
        end
        % 噪声模式：高斯 或 开关
        if isfield(option_BPNN, 'noise_mode')
            option_BPNN.noise_mode = option_BPNN.noise_mode;
        else
            option_BPNN.noise_mode = 'On_Off';
        end
        switch option_BPNN.noise_mode
            case 'Guass'
                if isfield(option_BPNN, 'noise_mean')
                    option_BPNN.noise_mean = option_BPNN.noise_mean;
                else
                    option_BPNN.noise_mean = 0;
                end
                if isfield(option_BPNN, 'noise_sigma')
                    option_BPNN.noise_sigma = option_BPNN.noise_sigma;
                else
                    option_BPNN.noise_sigma = 0.01;
                end
        end
    end
else
    option_BPNN.is_denoising = 0;
end

end