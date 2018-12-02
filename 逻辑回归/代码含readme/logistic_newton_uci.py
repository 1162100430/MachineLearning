from numpy import *
import numpy as np
import matplotlib.pyplot as pt


# 从文档中读取数据
def readdata():
    filename = 'uci.txt'
    X = []  # 属性的数组
    Y = []  # 类别（用0或者1表示）
    read = open(filename)
    for line in read.readlines():
        readin = line.strip().split()
        size = len(readin)
        temp = []
        temp.append(1.0)
        for i in range(size - 1):
            temp.append(float(readin[i]))
        X.append(temp)
        Y.append(int(readin[size - 1]))
    return X, Y


def sigmoid(d):
    # k = np.array(d)
    return 1.0 / (1.0 + exp(-d))


def newton(X, Y):
    x = mat(X)
    y = mat(Y).T
    m, n = shape(x)

    beta = mat(zeros(n)).T

    loop_max = 10000
    count = 0
    esil = 1e-6

    loss0 = float('inf')
    while count < loop_max:
        try:
            p = sigmoid(x * beta)
            # print(p-y)
            nabla = 1.0 / n * (x.T * (p - y))
            H = 1.0 / n * x.T * diag(p.getA1()) * diag((1 - p).getA1()) * x
            loss = 1.0 / n * sum(-y.getA1() * log(p.getA1()) - (1 - y).getA1()
                                 *log((1 - p).getA1()))
            beta = beta - H.I * nabla
            count += 1
            # print(count)
            if abs(loss0 - loss) < esil:
                break
            loss0 = loss
        except:
            H = H + 0.0001
            break
    return beta

def newton_regular(X,Y):
    x = mat(X)
    y = mat(Y).T
    m, n = shape(x)

    beta = mat(zeros(n)).T

    #设置正则化参数
    lanmuda = exp(-1)
    #设置迭代参数
    loop_max = 10000
    count = 0
    esil = 1e-6

    loss0 = float('inf')
    while count < loop_max:
        try:
            p = sigmoid(x * beta)
            # print(p-y)
            nabla = 1.0 / n * (x.T * (p - y))
            H = 1.0 / n * x.T * diag(p.getA1()) * diag((1 - p).getA1()) * x
            loss = 1.0 / n * sum(-y.getA1() * log(p.getA1()) - (1 - y).getA1() *log((1 - p).getA1()))
            beta = beta - H.I * nabla - beta*lanmuda
            count += 1
            # print(count)
            if abs(loss0 - loss) < esil:
                break
            loss0 = loss
        except:
            H = H + 0.0001
            break
    return beta

def accuracy(X, Y, weights):
    count = 0
    dataMat = mat(X)
    labelMat = mat(Y).T
    m, n = shape(dataMat)
    weights = mat(weights).T

    tp=0
    fn=0
    for i in range(m):
        h = sigmoid(np.dot(weights, (dataMat[i].T)))
        if(h>0.5 and int(labelMat[i, 0]) == 1):
            tp+=1
        elif(h>0.5 and int(labelMat[i, 0]) == 0):
            fn+=1

        if (h > 0.5 and int(labelMat[i, 0]) == 1) or (h < 0.5 and int(labelMat[i, 0]) == 0):
            count += 1
    return count / m,tp/(tp+fn)


def drawfigure(weights,title):
    dataMat, labelMat = readdata()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = pt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = arange(2.0, 5.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    pt.xlabel('X1')
    pt.ylabel('X2')
    pt.title(title)
    pt.show()

#画出测试集的图像
def drawtestfigure(weights,dataMat, labelMat,ifregular):
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = pt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='blue')
    x = arange(2.0, 4.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    pt.xlabel('X1')
    pt.ylabel('X2')
    pt.title(ifregular)
    pt.show()


if __name__ == '__main__':
    X, Y = readdata()
    x = []
    y = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(50):
        temp = []
        temp.append(X[i][0])
        temp.append(X[i][1])
        temp.append(X[i][2])
        if (i >= 40):
            x1.append(temp)
            y1.append(Y[i])
        else:
            x.append(temp)
            y.append(Y[i])
    p = 50
    while (p < 100):
        temp = []
        temp.append(X[p][0])
        temp.append(X[p][1])
        temp.append(X[p][2])
        if (p >= 90):
            x1.append(temp)
            y1.append(Y[p])
        else:
            x.append(temp)
            y.append(Y[p])
        p += 1
    theta1 = newton_regular(x, y)
    theta2 = newton(x, y)
    theta1 = array(theta1)
    theta2 = array(theta2)
    acc1, recall_regular = accuracy(x, y, theta1)
    acc2, recall = accuracy(x, y, theta2)

    #打印得到的参数
    print("有正则的参数W:"+str(theta1))
    print("无正则的参数W:"+str(theta2))

    print("训练集有正则准确率:" + str(acc1))
    print("训练集有正则召回率:" + str(recall_regular))
    print("训练集无正则准确率:" + str(acc2))
    print("训练集无正则召回率:"+str(recall))

    acc_test1, recall_test1 = accuracy(x1, y1, theta1)
    acc_test2, recall_test2 = accuracy(x1, y1, theta2)
    print("测试集无正则准确率:" + str(acc_test1))
    print("测试集无正则召回率:"+str(recall_test1))
    print("测试集有正则准确率:" + str(acc_test2))
    print("测试集有正则召回率:" + str(recall_test2))

    #画图
    drawfigure(theta1,'training_no_regular')
    drawfigure(theta2,'training_with_regular')
    drawtestfigure(theta1, x1, y1,'test_no_regular')
    drawtestfigure(theta2, x1, y1,'test_with_regular')
