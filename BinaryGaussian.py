import numpy as np
from svmutil import *
import sys

train_file = sys.argv[1]
test_file = sys.argv[2]

def GaussianKernel():
    clf = SVM(kernel=gaussian_kernel, C=1.0)
    clf.fit(subx_train, suby_train)

    train_pred = clf.predict(subx_train)
    train_acc = (train_pred == suby_train).sum()/len(suby_train)
    test_pred = clf.predict(subx_test)
    test_acc = (test_pred == suby_test).sum()/len(suby_test)
    #val_pred = clf.predict(subx_val)
    #val_acc = (val_pred == suby_val).sum()/len(suby_val)

    print("Train Accuracy = {}".format(train_acc))
    print("Test Accuracy = {}".format(test_acc))
    #print("Val Accuracy = {}".format(val_acc))
    ConfusionMatrix(suby_test, test_pred, 2, 'BinaryConfusionMatrix.png')

split = (8, 9)
x_train, y_train, m_train = getData('../fmnist_data/fashion_mnist/train.csv')
x_test, y_test, m_test = getData('../fmnist_data/fashion_mnist/test.csv')
#x_val, y_val, m_val = getData('../fmnist_data/fashion_mnist/test.csv')
print("Data Loaded")
indices_train = np.arange(x_train.shape[0])
indices_test = np.arange(x_test.shape[0])
#indices_val = np.arange(x_val.shape[0])
subx_train, suby_train = getXY(x_train, y_train, indices_train, split)
subx_test, suby_test = getXY(x_test, y_test, indices_test, split)
#subx_val, suby_val = getXY(x_val, y_val, indices_val, split)

GaussianKernel()