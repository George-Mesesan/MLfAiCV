from train.util import readData
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

trainImages, trainLabels = readData('train', 60000)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=15)
clf.fit(trainImages, trainLabels)

img = np.empty(28*28)
img.fill(255)

pixels = clf.tree_.feature
for i in range(0, len(pixels)):
    if pixels[i] > 0:
        img[pixels] = 0

disp_img = np.resize(img, [28, 28])
plt.imshow(disp_img, cmap='binary')
plt.show()