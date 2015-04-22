from train.util import readData
from sklearn import ensemble, tree, cross_validation

trainImages, trainLabels = readData('train', 60000)

tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=15)
scores = cross_validation.cross_val_score(tree, trainImages, trainLabels, cv=5)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

forest = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=20, max_depth=15)
scores = cross_validation.cross_val_score(forest, trainImages, trainLabels, cv=5)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))