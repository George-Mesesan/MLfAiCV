import numpy as np

imageSize = 28*28
filesDir = '../../'

def readImages(fileName, count):
    images = []
    with open(filesDir + fileName, 'rb') as fileIn:
        fileIn.seek(16)
        for i in range(0, count):
            imageBytes = fileIn.read(imageSize)
            images.append(np.frombuffer(imageBytes, dtype='b'))
    return images

def readLabels(fileName, count):
    with open(filesDir + fileName, 'rb') as fileIn:
        fileIn.seek(8)
        return np.frombuffer(fileIn.read(count), 'b')

def readData(filePrefix, count):
    images = readImages(filePrefix + '-images-idx3-ubyte', count)
    labels = readLabels(filePrefix + '-labels-idx1-ubyte', count)
    return images, labels

def classify(classifier, images, labels):
    correct = 0
    for i in range(0, len(labels)):
        label = classifier.predict(images[i])
        if label[0] == labels[i]:
            correct = correct + 1    
    print(correct/len(labels))  

def train(classifier, trainImages, trainLabels, testImages, testLabels):
    classifier.fit(trainImages, trainLabels)
    print(classifier)

    classify(classifier, trainImages, trainLabels)
    classify(classifier, testImages, testLabels)  