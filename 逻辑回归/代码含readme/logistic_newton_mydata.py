from numpy import *
import numpy as np
import matplotlib.pyplot as pt


# 从文档中读取数据
def readdata():
    filename = 'mydata.txt'
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
    x = arange(-2.0, 4.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    pt.xlabel('X1')
    pt.ylabel('X2')
    pt.title(title)
    pt.show()

def drawtestfigure(weights,dataMat, labelMat,title):
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
    x = arange(-2.0, 4.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    pt.xlabel('X1')
    pt.ylabel('X2')
    pt.title(title)
    pt.show()

def readtestdata():
    filename = 'mytest.txt'
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

def makedata():
    n=100
    co=0
    x1 = np.random.multivariate_normal([0, 1], [[0.5, co], [co, 0.5]], n)
    x2 = np.random.multivariate_normal([1,3],[[0.5,co],[co,0.5]],n)
    # si = np.vstack((x1, x2)).astype(np.float32)
    # si_labels = np.hstack((np.zeros(n), np.ones(n)))
    file = open('mydata_mu.txt', mode='w')
    for i in range(n):
        file.writelines(str(x1[i][0]))
        file.writelines('  ')
        file.writelines(str(x1[i][1]))
        file.writelines('  ')
        file.writelines('0')
        file.writelines('\n')

    for i in range(n):
        file.writelines(str(x2[i][0]))
        file.writelines('  ')
        file.writelines(str(x2[i][1]))
        file.writelines('  ')
        file.writelines('1')
        file.writelines('\n')
    file.close()

def maketestdata():
    n = 20
    co=0
    x1 = np.random.multivariate_normal([0, 1], [[0.5, co], [co, 0.5]], n)
    x2 = np.random.multivariate_normal([1,3],[[0.5,co],[co,0.5]],n)
    si = np.vstack((x1, x2)).astype(np.float32)
    si_labels = np.hstack((np.zeros(n), np.ones(n)))
    file = open('mytest_mu.txt', mode='w')
    for i in range(n):
        file.writelines(str(x1[i][0]))
        file.writelines('  ')
        file.writelines(str(x1[i][1]))
        file.writelines('  ')
        file.writelines('0')
        file.writelines('\n')

    for i in range(n):
        file.writelines(str(x2[i][0]))
        file.writelines('  ')
        file.writelines(str(x2[i][1]))
        file.writelines('  ')
        file.writelines('1')
        file.writelines('\n')
    file.close()

if __name__ == '__main__':
    # makedata()
    X, Y = readdata()
    theta1 = newton_regular(X, Y)
    theta2 = newton(X,Y)
    theta1 = array(theta1)
    theta2 = array(theta2)
    acc1,recall_regular = accuracy(X, Y, theta1)
    acc2,recall = accuracy(X, Y, theta2)

    #打印得到的参数
    print("有正则的参数W:"+str(theta1))
    print("无正则的参数W:"+str(theta2))

    print("训练集有正则准确率:" + str(acc1))
    print("训练集有正则召回率:" + str(recall_regular))
    print("训练集无正则准确率:" + str(acc2))
    print("训练集无正则召回率:"+str(recall))

    # maketestdata()
    si, sis = readtestdata()
    acc_test1,recall_test1 = accuracy(si, sis, theta1)
    acc_test2, recall_test2 = accuracy(si, sis, theta2)
    print("测试集有正则准确率:" + str(acc_test1))
    print("测试集有正则召回率:"+str(recall_test1))
    print("测试集无正则准确率:" + str(acc_test2))
    print("测试集无正则召回率:"+str(recall_test2))

    #画图
    drawfigure(theta1,'training_with_regular')
    drawfigure(theta2,'training_no_regular')
    drawtestfigure(theta1, si, sis,'test_with_regular')
    drawtestfigure(theta2, si, sis,'test_no_regular')
