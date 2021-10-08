import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from svmutil import *
import time

train_file = sys.argv[1]
test_file = sys.argv[2]

def kFoldCV(x_train, y_train, x_test, y_test):
	x = x_train
	y = y_train[:,0]
	C = [10, 5, 1, 1e-3, 1e-5]
	kFoldTrainAcc = []
	kFoldValAcc = []
	kFoldTestAcc = []
	for c in C:
		print("C = {}".format(c))
		clf = svm.SVC(C=c, kernel='rbf', gamma=0.05)
		clf_results = cross_validate(clf, x, y, cv=5, n_jobs=-1, return_train_score=True, return_estimator=True)
		train_acc = clf_results['train_score']
		val_acc = clf_results['test_score']
		test_acc = np.array([est.score(x_test, y_test[:,0]) for est in clf_results['estimator']])
		print("Train Acc = ", train_acc)
		print("Val Acc = ", val_acc)
		print("Test Acc = ", test_acc)
		kFoldTrainAcc.append(train_acc.sum()/5.0)
		kFoldValAcc.append(val_acc.sum()/5.0)
		kFoldTestAcc.append(test_acc.sum()/5.0)

	logC = np.log10(C)
	print("logC = ", logC)
	print("kFoldTrainAcc = ", kFoldTrainAcc)
	print("kFoldValAcc = ", kFoldValAcc)
	print("kFoldTestAcc = ", kFoldTestAcc)
	plt.figure()
	plt.plot(logC, kFoldTestAcc, label='Test Accuracy')
	plt.plot(logC, kFoldValAcc, label='Validation Accuracy')
	plt.title('kFold Cross Validation')
	plt.xlabel('logC')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('kFoldCrossValidation.png')
	#plt.show()
	plt.close()

x_train, y_train, m_train = getData(train_file)
x_test, y_test, m_test = getData(test_file)
print("Data Loaded")
start = time.time()
kFoldCV(x_train, y_train, x_test, y_test)
end = time.time()
print("Time = ", (end-start))