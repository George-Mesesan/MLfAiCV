from train.util import readData, train
from sklearn import svm

trainImages, trainLabels = readData('train', 2000)
testImages, testLabels = readData('t10k', 10000)

train(svm.SVC(kernel='linear'), trainImages, trainLabels, testImages, testLabels)
