n1 = 80; n2 = 40;
S1 = eye(2); S2 = [1 0.95; 0.95 1];
m1 = [0.75; 0]; m2 = [-0.75; 0];

x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);         

x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
classifier = fitensemble(x, y, 'Bag', 1000, 'Tree', 'Type', 'classification');

testn1 = 500; testn2 = 500;
testX1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, testn1), m1);
testX2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, testn2), m2);
testX = [testX1 testX2];
testY = [-ones(1,testn1) ones(1,testn2)];

[predictY lp2] = predict(classifier, testX');

sum(predictY == testY') / length(predictY)
correct = predictY == testY';
u_correct = uncertainty([lp2(correct(1:testn1),1); lp2(correct(testn1+1:end),2)]);
figure(9);
histogram(u_correct, 10, 'BinLimits', [0 1]);
title('Uncertainty histogram for correctly classified data');

wrong = not(correct);
u_wrong = uncertainty([lp2(wrong(1:testn1),1); lp2(wrong(testn1+1:end),2)]);
figure(10);
histogram(u_wrong, 10, 'BinLimits', [0 1]);
title('Uncertainty histogram for misclassified data');