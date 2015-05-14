show_plot = 0;
n1 = 80; n2 = 40;
S1 = eye(2); S2 = [1 0.95; 0.95 1];
m1 = [0.75; 0]; m2 = [-0.75; 0];

x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);         

x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';

[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs
if show_plot
    figure(6)
    plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
    plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12);
    tmm = bsxfun(@minus, t, m1');
    p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
    tmm = bsxfun(@minus, t, m2');
    p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
    set(gca, 'FontSize', 24)
    colormap(jet);
    contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9])
    colorbar
    grid
    axis([-4 4 -4 4])
end

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   hyp.cov = log([1 1 1]);
likfunc = @likLogistic;
inffunc = @infLaplace;

hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfunc, likfunc, x, y);
[a b c d lp] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
if show_plot
    figure(7)
    set(gca, 'FontSize', 24)
    plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
    plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
    colormap(jet);
    contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
    colorbar
    grid
    axis([-4 4 -4 4])
end

testn1 = 500; testn2 = 500;
testX1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, testn1), m1);
testX2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, testn2), m2);
testX = [testX1 testX2];
testY = [-ones(1,testn1) ones(1,testn2)];
[a2 b2 c2 d2 lp2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, testX', testY);

if show_plot
    figure(8)
    set(gca, 'FontSize', 24)
    plot(testX1(1,:), testX1(2,:), 'b+', 'MarkerSize', 12); hold on
    plot(testX2(1,:), testX2(2,:), 'r+', 'MarkerSize', 12)
    colormap(jet);
    contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
    colorbar
    grid
    axis([-4 4 -4 4])
end

sum(lp2 > log(0.5)) / length(lp2)
thresh = log(0.5);
u_correct = uncertainty(exp(lp2(lp2 > thresh)));
figure(9);
histogram(u_correct, 10, 'BinLimits', [0 1]);
title('Uncertainty histogram for correctly classified data');

u_wrong = uncertainty(exp(lp2(lp2 <= thresh)));
figure(10);
histogram(u_wrong, 10, 'BinLimits', [0 1]);
title('Uncertainty histogram for misclassified data');