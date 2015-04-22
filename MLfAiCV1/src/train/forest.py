from train.util import readData, train
from sklearn import ensemble

trainImages, trainLabels = readData('train', 60000)
testImages, testLabels = readData('t10k', 10000)

train(ensemble.RandomForestClassifier(criterion='entropy', n_estimators=20, max_depth=15), trainImages, trainLabels, testImages, testLabels)
