function option_SAE = get_SAE_option(preOption_SAE, varargin)
%设置SAE的参数
% by 郑煜伟 Ewing 2016-04
% 输入：SAE网络的选项 preOption_SAE
% 返回：SAE网络的选项 option_SAE

if exist('preOption_SAE', 'var')
    % 得到AE的一些预选参数
    if isfield(preOption_SAE, 'option_AE')
        option_SAE.option_AE = get_AE_option(preOption_SAE.option_AE); 
    else
        option_SAE.option_AE = get_AE_option([]);
    end
    % 得到BP的一些预选参数
    if isfield(preOption_SAE, 'option_BP')
        option_SAE.option_BP = get_BP_option(preOption_SAE.option_BP);
    else
        option_SAE.option_BP = get_BP_option([]);
    end
else
    option_SAE.option_AE = get_AE_option([]); % 得到AE的一些预选参数
    option_SAE.option_BP = get_BP_option([]); % 得到BP的一些预选参数
end

end


function option_AE = get_AE_option(preOption_AE)
%设置AE的参数
% 输入 AE网络的选项 preOption_AE
% 返回：
% AE网络的选项：option_AE
% decay_lambda：		权重衰减系数——正则项罚项权重；
% activation：		激活函数类型：sigmoid，ReLU，weakly_ReLU，tanh激活函数类型：sigmoid，ReLU，weakly_ReLU，tanh
% slope：			激活函数为weakly_ReLU时，负方向的斜率，默认0.2；

% is_batch_norm：	是否使用 Batch Normalization 来 speed-up学习速度；

% is_sparse：		是否使用 sparse hidden level 的规则；
% sparse_rho：		稀疏性中rho；
% sparse_beta：		稀疏性罚项权重；

% is_denoising：		是否使用 denoising 规则
% noise_layer：		AE中添加噪声的层：'first_layer' or 'all_layers'
% noise_rate：		每一位添加噪声的概率
% noise_mode：		添加噪声的模式：'On_Off' or 'Guass'
% noise_mean：		高斯模式：均值
% noise_sigma：		高斯模式：标准差

% is_weighted_cost：	是否对每一位数据的cost进行加权对待
% weighted_cost：	加权cost的权重

    if isfield(preOption_AE, 'decay_lambda')
        option_AE.decay_lambda = preOption_AE.decay_lambda;
    else
        option_AE.decay_lambda = 0.01;
    end
    if isfield(preOption_AE, 'activation')
        option_AE.activation = preOption_AE.activation;
		if strcmp(option_AE.activation{:}, 'weakly_ReLU')
			if isfield( preOption_AE, 'slope' )
				option_AE.slope = preOption_AE.slope;
			else
				option_AE.slope = 0.2;
			end
		end
    else
        option_AE.activation = { 'sigmoid' };
    end

    % batchNorm
    if isfield(preOption_AE, 'is_batch_norm')
        option_AE.is_batch_norm = preOption_AE.is_batch_norm;
    else
        option_AE.is_batch_norm = 0;
    end

    % sparse
    if isfield(preOption_AE, 'is_sparse')
        option_AE.is_sparse = preOption_AE.is_sparse;
    else
        option_AE.is_sparse = 0;
    end
    if option_AE.is_sparse
        if isfield(preOption_AE, 'sparse_rho')
            option_AE.sparse_rho = preOption_AE.sparse_rho;
        else
            option_AE.sparse_rho = 0.1;
        end
        if isfield(preOption_AE, 'sparse_beta')
            option_AE.sparse_beta = preOption_AE.sparse_beta;
        else
            option_AE.sparse_beta = 0.3;
        end
    end

    % de-noising
    if isfield(preOption_AE, 'is_denoising')
        option_AE.is_denoising = preOption_AE.is_denoising;
		if option_AE.is_denoising
			% de-noising每一层 或 只第一个输入层
			if isfield(preOption_AE, 'noise_layer')
				option_AE.noise_layer = preOption_AE.noise_layer;
			else
				option_AE.noise_layer = 'first_layer';
			end
			% 噪声概率
			if isfield(preOption_AE, 'noise_rate')
				option_AE.noise_rate = preOption_AE.noise_rate;
			else
				option_AE.noise_rate = 0.1;
			end
			% 噪声模式：高斯 或 开关
			if isfield(preOption_AE, 'noise_mode')
				option_AE.noise_mode = preOption_AE.noise_mode;
			else
				option_AE.noise_mode = 'On_Off';
			end
			switch option_AE.noise_mode
				case 'Guass'
					if isfield(preOption_AE, 'noise_mean')
						option_AE.noise_mean = preOption_AE.noise_mean;
					else
						option_AE.noise_mean = 0;
					end
					if isfield(preOption_AE, 'noise_sigma')
						option_AE.noise_sigma = preOption_AE.noise_sigma;
					else
						option_AE.noise_sigma = 0.01;
					end
			end
		end
    else
        option_AE.is_denoising = 0;
    end

    % weighted_cost
    if isfield(preOption_AE, 'is_weighted_cost')
        option_AE.is_weighted_cost = preOption_AE.is_weighted_cost;
    else
        option_AE.is_weighted_cost = 0;
    end
    if option_AE.is_weighted_cost
        if isfield(preOption_AE, 'weighted_cost')
            option_AE.weighted_cost = preOption_AE.weighted_cost;
%         else
%             error( '加权cost一定要自己设置权重向量！' );
        end
    end
end


function option_BP = get_BP_option(preOption_BP)
%设置BP的参数
% 输入 BP网络的选项 preOption_BP
% 返回：
% AE网络的选项：option_BP
% decay_lambda：	权重衰减系数——正则项罚项权重；
% activation：	激活函数类型；

% is_batch_norm：是否使用 Batch Normalization 来 speed-up学习速度；

% is_denoising：	是否使用 denoising 规则
% noise_layer：	AE中添加噪声的层：'first_layer' or 'all_layers'
% noise_rate：	每一位添加噪声的概率
% noise_mode：	添加噪声的模式：'On_Off' or 'Guass'
% noise_mean：	高斯模式：均值
% noise_sigma：	高斯模式：标准差

    if isfield(preOption_BP, 'decay_lambda')
        option_BP.decay_lambda = preOption_BP.decay_lambda;
    else
        option_BP.decay_lambda = 0.001;
    end
    if isfield(preOption_BP, 'activation')
        option_BP.activation = preOption_BP.activation;
    else
        option_BP.activation = {'softmax'};
    end

    % batch normalization
    if isfield(preOption_BP, 'is_batch_norm')
        option_BP.is_batch_norm = preOption_BP.is_batch_norm;
    else
        option_BP.is_batch_norm = 0;
    end

    % de-noising
    if isfield(preOption_BP, 'is_denoising')
        option_BP.is_denoising = preOption_BP.is_denoising;
		if option_BP.is_denoising
			% denoising每一层 或 只第一个输入层
			if isfield(preOption_BP, 'noise_layer')
				option_BP.noise_layer = preOption_BP.noise_layer;
			else
				option_BP.noise_layer = 'first_layer';
			end
			% 噪声概率
			if isfield(preOption_BP, 'noise_rate')
				option_BP.noise_rate = preOption_BP.noise_rate;
			else
				option_BP.noise_rate = 0.1;
			end
			% 噪声模式：高斯 或 开关
			if isfield(preOption_BP, 'noise_mode')
				option_BP.noise_mode = preOption_BP.noise_mode;
			else
				option_BP.noise_mode = 'OnOff';
			end
			switch option_BP.noise_mode
				case 'Guass'
					if isfield(preOption_BP, 'noise_mean')
						option_BP.noise_mean = preOption_BP.noise_mean;
					else
						option_BP.noise_mean = 0;
					end
					if isfield(preOption_BP, 'noise_sigma')
						option_BP.noise_sigma = preOption_BP.noise_sigma;
					else
						option_BP.noise_sigma = 0.01;
					end
			end
		end
    else
        option_BP.is_denoising = 0;
    end
end




