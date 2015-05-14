function [ correct ] = train_predict( train_X, train_Y, test_X, test_Y, boost_type, learner)

classifier = fitensemble(train_X, train_Y, boost_type, 1000, learner);
predict_Y = predict(classifier, test_X);
correct = predict_Y == test_Y;
sum(correct)/length(correct)

end

