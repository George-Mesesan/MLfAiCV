from train.util import readData
from sklearn import ensemble
import numpy as np
import matplotlib.pyplot as plt

trainImages, trainLabels = readData('train', 60000)
clf = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=20, max_depth=15)
clf.fit(trainImages, trainLabels)

img = np.empty(28*28)
img.fill(255)

estimators = clf.estimators_
for i in range(0, len(estimators)):
    pixels = estimators[i].tree_.feature
    for j in range(0, len(pixels)):
        if pixels[j] > 0:
            img[pixels] = 0

disp_img = np.resize(img, [28, 28])
plt.imshow(disp_img, cmap='binary')
plt.show()