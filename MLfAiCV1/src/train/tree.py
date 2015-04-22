from train.util import readData, train
from sklearn import tree

trainImages, trainLabels = readData('train', 60000)
testImages, testLabels = readData('t10k', 10000)

train(tree.DecisionTreeClassifier(criterion='entropy', max_depth=15), trainImages, trainLabels, testImages, testLabels)
