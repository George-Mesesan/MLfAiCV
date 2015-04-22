from train.util import readData
from sklearn import tree

trainImages, trainLabels = readData('train', 60000)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
clf.fit(trainImages, trainLabels)

with open("../../viz/tree.dot", 'w') as f:
    tree.export_graphviz(clf, out_file=f)
    