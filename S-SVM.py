import matplotlib.pyplot as plt
import random
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, model_selection
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np


def loadDataSet():
    dataMat = []  # 列表list
    labelMat = []
    txt = open('./Data/LR/horse.txt')
    for line in txt.readlines():
        lineArr = line.strip().split()  # strip():返回一个带前导和尾随空格的字符串的副本
        # split():默认以空格为分隔符，空字符串从结果中删除
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        if int(lineArr[2])==0:
            labelMat.append(-1)
        else:
            labelMat.append(int(lineArr[2]))
    x_train = dataMat[:70]
    y_train = labelMat[:70]
    x_test = dataMat[70:]
    y_test = labelMat[70:]

    np.save("./Data/s-svm/train_data.npy", np.array(x_train))
    np.save("./Data/s-svm/train_target.npy", np.array(y_train))
    np.save("./Data/s-svm/test_data.npy", np.array(x_test))
    np.save("./Data/s-svm/test_target.npy", np.array(y_test))
    return x_train, y_train, x_test, y_test


class LinearSVM:
    def __init__(self):
        self.w = self.b = None

    def fit(self, x, y, c=1, lr=0.01, epoch=10000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        self.w = np.zeros(x.shape[1])
        self.b = 2
        for _ in range(epoch):
            self.w *= 1 - lr
            error = 1 - y * self.predict(x, True)
            idx = np.argmax(error)
            if error[idx] <= 0:
                continue
            delta = lr * c * y[idx]
            self.w += delta * x[idx]
            self.b += delta

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        y_pred = x.dot(self.w) + self.b
        if raw:
            return y_pred
        return np.sign(y_pred).asstype(np.float32)


def showClassifer(x_train, y_train, w, b):
    # 绘制样本点
    dataMat = np.mat(x_train).tolist()
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        if np.mat(y_train).T[i, 0] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7, c='r')  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7, c='g')  # 负样本散点图
    # 绘制直线
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])

    plt.show()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = loadDataSet()
    t = LinearSVM()
    t.fit(x=x_train, y=y_train, c=1, lr=0.01, epoch=10000)
    w, b = t.w, t.b
    print(w, b)
    showClassifer(x_train, y_train, w, b)
