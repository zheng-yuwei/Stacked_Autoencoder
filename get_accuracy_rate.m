function accuracy = get_accuracy_rate( predict_labels, labels )
%计算预测准确率
% by 郑煜伟 Ewing 2016-04

% 将预测的概率矩阵中，每列最大概率的值置1，其他置0
predict_labels = bsxfun(@eq, predict_labels, max( predict_labels ));
predict_labels(:, sum(predict_labels)>1) = 0;  % 多个值相等则应都被置0，也就是不正确
% 找出正确label所对应矩阵的位置，并对这些位置的值求均值
index_row = labels';
index_col = 1:length(index_row);
index     = (index_col - 1) .* size( predict_labels, 1 ) + index_row;
accuracy  = sum( predict_labels(index) )/length(index_row);

end