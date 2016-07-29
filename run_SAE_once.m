function [opt_theta, accuracy] = run_SAE_once(images_Train, labels_Train, ...
    images_Test, labels_Test, ... % 数据
    architecture, ...
    option_SAE, option_BPNN, ...
    is_disp_network, is_disp_info)
%设置SAE参数 并 运行一次 SAE
% by 郑煜伟 Ewing 2016-04

%% 训练SAE
theta_SAE = train_SAE(images_Train, labels_Train, architecture, option_SAE); % 训练SAE
if is_disp_network
    % 展示网络中间层所抽取的feature
    display_network(reshape(theta_SAE(1 : 784 * 400), 400, 784));
    display_network((reshape(theta_SAE(1 : 784 * 400), 400, 784)' * ...
        reshape(theta_SAE(784 * 400 + 1 : 784 * 400 + 400*200 ), 200, 400)')');
end
if is_disp_info
    % 用 未微调的SAE参数 进行预测
    predict_labels = predictNN(images_Train, architecture, theta_SAE, option_BPNN);
    accuracy = get_accuracy_rate(predict_labels, labels_Train);
    disp(['MNIST训练集 SAE(未微调）准确率为： ', num2str(accuracy * 100), '%']);
    
    predict_labels = predictNN( images_Test, architecture, theta_SAE, option_BPNN );
    accuracy = get_accuracy_rate( predict_labels, labels_Test );
    disp(['MNIST测试集 SAE(未微调）准确率为： ', num2str(accuracy * 100), '%']);
end

%% BP fine-tune
[opt_theta, ~] = train_BPNN( images_Train, labels_Train, theta_SAE, architecture, option_BPNN );
if is_disp_network
    % 展示网络中间层所抽取的feature
    display_network(reshape(opt_theta(1 : 784 * 400), 400, 784) );
    display_network((reshape(opt_theta(1 : 400 * 784), 400, 784)' * ...
        reshape(opt_theta(784 * 400 + 1 : 784 * 400 + 400*200 ), 200, 400)')' );
end
%% 用 fine-tune后SAE 进行预测
if is_disp_info
    predict_labels = predict_NN(images_Train, architecture, opt_theta, option_BPNN);
    accuracy = get_accuracy_rate(predict_labels, labels_Train);
    disp(['MNIST训练集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%']);
end
predict_labels = predict_NN( images_Train, architecture, opt_theta, option_BPNN);
accuracy = get_accuracy_rate( predict_labels, labels_Train );
disp(['MNIST训练集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%'] );
predict_labels = predict_NN( images_Test, architecture, opt_theta, option_BPNN);
accuracy = get_accuracy_rate( predict_labels, labels_Test );
disp(['MNIST测试集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%']);% pppppppppppppppppppppp
if is_disp_info
    disp(['MNIST测试集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%']);
end

end


