from train.util import readData, train
from sklearn import svm

trainImages, trainLabels = readData('train', 60000)
testImages, testLabels = readData('t10k', 10000)

train(svm.SVC(kernel='rbf'), trainImages, trainLabels, testImages, testLabels)
train(svm.SVC(kernel='linear'), trainImages, trainLabels, testImages, testLabels)
train(svm.SVC(kernel='poly'), trainImages, trainLabels, testImages, testLabels)
train(svm.LinearSVC(), trainImages, trainLabels, testImages, testLabels)