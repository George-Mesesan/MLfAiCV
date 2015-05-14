[train_X, train_Y] = read_data('train');
%train_X = train_X(1:1000,:);
%train_Y = train_Y(1:1000,:);

[test_X, test_Y] = read_data('t10k');

classifier = fitensemble(train_X, train_Y, 'AdaBoostM2', 1000, 'Tree');
rsLoss = resubLoss(classifier, 'Mode', 'Cumulative');
plot(rsLoss);

predict_Y = predict(classifier, test_X);
correct = predict_Y == test_Y;
sum(correct)/length(correct)

train_predict(train_X, train_Y, test_X, test_Y, 'LPBoost', 'Tree');
train_predict(train_X, train_Y, test_X, test_Y, 'TotalBoost', 'Tree');
train_predict(train_X, train_Y, test_X, test_Y, 'RUSBoost', 'Tree');
train_predict(train_X, train_Y, test_X, test_Y, 'Subspace', 'Discriminant');
train_predict(train_X, train_Y, test_X, test_Y, 'Subspace', 'KNN');

classifier = fitensemble(train_X, train_Y, 'Bag', 1000, 'Tree', 'Type', 'classification');
predict_Y = predict(classifier, test_X);
correct = predict_Y == test_Y;
sum(correct)/length(correct)
