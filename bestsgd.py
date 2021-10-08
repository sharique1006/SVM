import numpy as np
import json
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import time
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

train_data = [json.loads(line) for line in open(train_file, 'r')]
test_data = [json.loads(line) for line in open(test_file, 'r')]
m_train = len(train_data)
m_test = len(test_data)

reviews_train = [train_data[i]['text'] for i in range(m_train)]
stars_train = [train_data[i]['stars'] for i in range(m_train)]
reviews_test = [test_data[i]['text'] for i in range(m_test)]
stars_test = [test_data[i]['stars'] for i in range(m_test)]

x_train, x_val, y_train, y_val = train_test_split(reviews_train, stars_train, test_size=0.1)
valAcc = []
classifiers = []

def Accuracy(clf, x, y):
	pred = clf.predict(x)
	acc = np.mean(y == pred)
	return pred, acc

def SGD():
	alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]
	for a in alpha:
		#print("alpha = ", a)
		clf = Pipeline([('vect', CountVectorizer()), ('clf', SGDClassifier(loss='log', alpha=a))])
		#start = time.time()
		clf.fit(x_train, y_train)
		#end = time.time()
		#train_pred, train_acc = Accuracy(clf, x_train, y_train)
		val_pred, val_acc = Accuracy(clf, x_val, y_val)
		valAcc.append(val_acc)
		classifiers.append(clf)
		#print("Training Time = ", end-start)
		#print("Training Accuracy = ", train_acc)
		#print("Val Accuracy = ", val_acc)
	bestvalAcc = np.argmax(np.array(valAcc))
	#print("Best Val Acc = ", bestvalAcc)
	bestclf = classifiers[bestvalAcc]
	bestalpha = alpha[bestvalAcc]
	#print("Best Alpha = ", bestalpha)
	return bestclf, bestalpha

clf, alpha = SGD()
test_pred, test_acc = Accuracy(clf, reviews_test, stars_test)
#print("Test Accuracy = ", test_acc)

f = open(output_file, 'w')
for pred in test_pred:
	print((int)(pred), file=f)
