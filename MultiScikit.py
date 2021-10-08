import numpy as np
from sklearn import svm
from svmutil import *
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]

def scikitSVM(x_train, y_train, x_test, y_test, C):
	x = x_train
	y = y_train[:,0]
	clf = svm.SVC(C=C, kernel='rbf', gamma=0.05)
	clf.fit(x, y)

	train_acc = clf.score(x, y)
	#val_acc = clf.score(x_val, y_val[:,0])
	test_acc = clf.score(x_test, y_test[:,0])
	
	print("Train Accuracy = {}".format(train_acc))
	#print("Val Accuracy = {}".format(val_acc))
	print("Test Accuracy = {}".format(test_acc))

x_train, y_train, m_train = getData(train_file)
x_test, y_test, m_test = getData(test_file)
#x_val, y_val, m_val = getData('../fmnist_data/fashion_mnist/test.csv')

scikitSVM(x_train, y_train, x_test, y_test, 1.0)