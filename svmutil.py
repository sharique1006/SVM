import numpy as np
import numexpr as ne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cvxopt

def getData(file):
    x = np.loadtxt(file, delimiter=',', dtype=np.float64)
    y = x[:,-1].reshape(-1,1)
    x = np.delete(x,-1,1)
    x /= 255.0
    m = len(x)
    return x, y, m

def getXY(x, y, indices, pair):
	pos1 = indices[(y == pair[0]).ravel()]
	pos2 = indices[(y == pair[1]).ravel()]
	pos = np.concatenate((pos1, pos2))
	subx = x[pos]
	suby = y[pos]
	suby = (suby == pair[0]).astype(np.float64) - (suby == pair[1]).astype(np.float64)
	suby = suby[:,0]
	return subx, suby

def linear_kernel(X, Y):
	return np.dot(X, Y.T)

def gaussian_kernel(X, Y):
    X_norm = -np.einsum('ij,ij->i', X, X)
    Y_norm = -np.einsum('ij,ij->i', Y, Y)
    Z = ne.evaluate('exp(g * (A + B + 2*C))',{
    	'A': X_norm[:,None],
        'B': Y_norm[None,:],
        'C': np.dot(X, Y.T),
        'g': 0.05
        })
    return Z

class SVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        m, n = X.shape
        K = self.kernel(X, X)
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(m) * -1)
        l = y.reshape(-1, 1) * 1.
        A = cvxopt.matrix(l.reshape(1,-1))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        LM = np.ravel(solution['x'])
        svi = LM > 1e-5
        ind = np.arange(len(LM))[svi]
        sm = LM[svi]
        sv = X[svi]
        svl = y[svi]

        b = 0
        for i in range(len(sm)):
            b += svl[i] - np.sum(sm * svl * K[ind[i], svi])
        b /= len(sm)

        if self.kernel == linear_kernel:
            w = ((svl * sm).T @ sv)
        else:
            w = None

        self.sm = sm
        self.sv = sv
        self.svl = svl
        self.b = b
        self.w = w

    def predict(self, X):
        if self.w is not None:
            prediction = np.dot(X, self.w) + self.b
        else:
            y_predict = self.kernel(X, self.sv) @ (self.sm * self.svl)
            prediction = y_predict + self.b
        return np.sign(prediction)

def ConfusionMatrix(y, pred, lb, file):
    classes = np.arange(0,lb,dtype=np.int64)
    confusion_matrix = np.zeros((classes.shape[0], classes.shape[0]), dtype=np.int64)
    for l, p in zip(y, pred):
        confusion_matrix[classes[int(p)], classes[int(l)]] += 1

    TP = confusion_matrix.trace()
    FP = confusion_matrix.sum(axis=0)
    FN = confusion_matrix.sum(axis=1)
    Precision = TP/(TP + FP)
    Recall = TP/(TP + FN)
    #print("True Positives = ", TP)
    #print("False Positives = ", FP)
    #print("False Negatives = ", FN)
    #print("Precision = ", Precision)
    #print("Recall = ", Recall)

    fig = plt.figure(figsize=(10,8))
    ax = fig.gca()
    _ = sns.heatmap(confusion_matrix, annot=True, cmap="Blues", xticklabels=classes, yticklabels=classes, fmt='g')
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Predicted Class")
    plt.title("Confusion Matrix", y=1.08)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.savefig(file)
    #plt.show()
    plt.close()