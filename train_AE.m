function [opt_theta, cost] = train_AE(input, theta, architecture, count_AE, option_AE)
%训练AE网络
% by 郑煜伟 Ewing 2016-04

% 函数 calc_AE_Batch 可以根据当前点计算 cost 和 gradient，但是步长不确定
% 这里，调用Mark Schmidt的包来优化迭代 步长：用了l-BFGS
% Mark Schmidt (http://www.di.ens.fr/~mschmidt/Software/minFunc.html) [仅供学术]
addpath minFunc/
options.Method = 'lbfgs'; % 其实不一定用L-BFGS，可以参考 On optimization methods for deep learning
options.maxIter = 100;	  % L-BFGS 的最大迭代代数
options.display = 'off';
% options.TolX = 1e-3;

% 判断该 countAE层 AE是否需要添加noise 以 使用denoising规则
[is_denoising, input_corrupted ] = denoising_switch(input, count_AE, option_AE);
if is_denoising
	[opt_theta, cost] = minFunc(@(x) calc_AE_batch(input, x, architecture, option_AE, input_corrupted), ...
            theta, options);
else
	[opt_theta, cost] = minFunc(@(x) calc_AE_batch(input, x, architecture, option_AE), ...
            theta, options);
end

end

function [is_denoising, input_corrupted] = denoising_switch(input, count_AE, option_AE)
%判断该层AE是否需要添加noise以使用denoising规则
% 返回 是否is_denoising的标志 及 噪声

% is_denoising：	是否使用 denoising 规则
% noise_layer：	AE中添加噪声的层：'first_layer' or 'all_layers'
% noise_rate：	每一位添加噪声的概率
% noise_mode：	添加噪声的模式：'On_Off' or 'Guass'
% noise_mean：	高斯模式：均值
% noise_sigma：	高斯模式：标准差

    is_denoising    = 0;
    input_corrupted = [];
    if option_AE.is_denoising
        switch option_AE.noise_layer
            case 'first_layer'
                if count_AE == 1
                    is_denoising = 1;
                end
            case 'all_layers'
                is_denoising = 1;
            otherwise
                error( '错误的AE噪声层数！' );
        end
        
        if is_denoising
            input_corrupted = input;
            index_corrupted = rand(size(input)) < option_AE.noise_rate;
            switch option_AE.noise_mode
                case 'Guass'
                    % 均值为 noise_mean，标准差为 noise_sigma 的高斯噪声
                    noise = option_AE.noise_mean + ...
                        randn(size(input)) * option_AE.noise_sigma;
                    noise(~index_corrupted) = 0;
                    input_corrupted = input_corrupted + noise;
                case 'On_Off'
                    input_corrupted(index_corrupted) = 0;
            end
        end
    end
end

