import numpy as np
import itertools
from scipy import stats
from svmutil import *
from sklearn import svm
from sklearn.model_selection import cross_validate
import time
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

def kFoldCV(x_train, y_train, x_test, y_test):
	x = x_train
	y = y_train[:,0]
	C = [10, 5, 1]
	kFoldValAcc = []
	for c in C:
		#print("C = {}".format(c))
		clf = svm.SVC(C=c, kernel='rbf', gamma=0.05)
		clf_results = cross_validate(clf, x, y, cv=5, n_jobs=-1, return_train_score=False, return_estimator=True)
		val_acc = clf_results['test_score']
		kFoldValAcc.append(val_acc.sum()/5.0)
	bestvalAcc = np.argmax(np.array(kFoldValAcc))
	bestC = C[bestvalAcc]
	#print("Best Value of C = ", bestC)
	return bestC

def OnevsOne(x, y, labels, C):
	classifiers = []
	labelPairs = list(itertools.combinations(np.arange(len(labels)), 2))
	indices = np.arange(x.shape[0])
	for pair in labelPairs:
		subx, suby = getXY(x, y, indices, pair)
		clf = SVM(kernel=gaussian_kernel, C=C)
		clf.fit(subx, suby)
		classifiers.append((clf, pair[0], pair[1]))
	return classifiers

def OnevsOnePredict(classifiers, x_test):
	score = np.zeros((x_test.shape[0], len(classifiers)))
	ones = np.ones((x_test.shape[0], 1))
	for i, clf in enumerate(classifiers):
		pred = clf[0].predict(x_test).reshape(-1,1)
		score[:,i] = np.where(pred == 1, clf[1]*ones, clf[2]*ones).ravel()
	prediction = -stats.mode(-score, axis=1)[0]
	return prediction

x_train, y_train, m_train = getData(train_file)
x_test, y_test, m_test = getData(test_file)
#x_val, y_val, m_val = getData('../fmnist_data/fashion_mnist/test.csv')
labels = np.unique(y_train)
num_labels = len(labels)
#print("Data Loaded")

C = kFoldCV(x_train, y_train, x_test, y_test)

#start = time.time()
classifiers = OnevsOne(x_train, y_train, labels, C)
#end = time.time()
#print("Training Time = {}".format(end-start))

#start = time.time()
#train_pred = OnevsOnePredict(classifiers, x_train)
#end = time.time()
#print("Training Prediction Time = {}".format(end-start))
#train_acc = (y_train == train_pred).sum()/len(y_train)
'''
start = time.time()
val_pred = OnevsOnePredict(classifiers, x_val)
end = time.time()
print("Validation Prediction Time = {}".format(end-start))
val_acc = (y_val == val_pred).sum()/len(y_val)
'''
#start = time.time()
test_pred = OnevsOnePredict(classifiers, x_test)
#end = time.time()
#print("Test Prediction Time = {}".format(end-start))
test_acc = (y_test == test_pred).sum()/len(y_test)

#print("Train Accuracy = ", train_acc)
#print("Val Accuracy = ", val_acc)
#print("Test Accuracy = ", test_acc)

z = test_pred[:,0]
f = open(output_file, 'w')
for pred in z:
	print((int)(pred), file=f)

#ConfusionMatrix(y_test, test_pred, num_labels, 'MultiConfusionMatrix.png')