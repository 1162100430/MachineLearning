# 利用自己生成的数据进行测试

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

def sigmoid(d):
    k = np.array(d)
    return 1.0 / (1 + exp(-k[0][0]))

#此方法用于生成训练集数据并写回到mydata.txt文件里面
def makedata():
    n=100
    co=1
    x1 = np.random.multivariate_normal([0, 1], [[0.5, co], [co, 0.5]], n)
    x2 = np.random.multivariate_normal([0,1],[[0.5,co],[co,0.5]],n)
    si = np.vstack((x1, x2)).astype(np.float32)
    si_labels = np.hstack((np.zeros(n), np.ones(n)))
    file = open('mydata.txt', mode='w')
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

#用于生成测试集数据
def maketestdata():
    n = 20
    co=1
    x1 = np.random.multivariate_normal([0, 1], [[0.5, co], [co, 0.5]], n)
    x2 = np.random.multivariate_normal([0,1],[[0.5,co],[co,0.5]],n)
    si = np.vstack((x1, x2)).astype(np.float32)
    si_labels = np.hstack((np.zeros(n), np.ones(n)))
    file = open('mytest.txt', mode='w')
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

#无正则化的梯度下降法
def GradientDescent(X, Y):
    x = mat(X)
    y = Y
    m, n = shape(x)

    # 设置终止条件
    loop_max = 100000
    epsil = 1e-4
    error = np.zeros(n)

    alpha = 0.002  # 设置步长
    count = 0

    # 初始化权值
    np.random.seed(0)
    theta = np.random.randn(n)

    while count < loop_max:
        count += 1
        # print(count)
        sum_m = np.zeros(n)
        for i in range(m):
            py1 = sigmoid(np.dot(theta, (x[i].T)))
            change = (y[i] - py1)
            p=np.array(x[i])
            # print(p[0])
            dif = change * p[0]
            sum_m = sum_m + dif

        theta = theta + alpha * sum_m

        #判断是否已经收敛
        if np.linalg.norm(theta - error) < epsil:
            break
        else:
            error = theta
    return theta

#带正则化的梯度下降
def GradientDescent_regular(X,Y):
    x = mat(X)
    y = Y
    m, n = shape(x)

    # 设置终止条件
    loop_max = 1000000
    epsil = 1e-4
    error = np.zeros(n)

    alpha = 0.002  # 设置步长
    lanmuda = exp(-1)
    count = 0

    # 初始化权值
    np.random.seed(0)
    theta = np.random.randn(n)

    while count < loop_max:
        count += 1
        # print(count)
        sum_m = np.zeros(n)
        for i in range(m):
            py1 = sigmoid(np.dot(theta, (x[i].T)))
            change = (y[i] - py1)
            p=np.array(x[i])
            # print(p[0])
            dif = change * p[0]
            sum_m = sum_m + dif

        theta = theta - (alpha*lanmuda)*theta+ alpha * sum_m

        #判断是否已经收敛
        if np.linalg.norm(theta - error) < epsil:
            break
        else:
            error = theta
    return theta

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
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    pt.xlabel('X1')
    pt.ylabel('X2')
    pt.title(title)
    pt.show()

#画出测试集的图像
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

def accuracy(X,Y,weights):
    count = 0
    dataMat = mat(X)
    labelMat = mat(Y).T
    m, n = shape(dataMat)
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

if __name__ == '__main__':
    # makedata()
    X, Y = readdata()
    theta1 = GradientDescent(X, Y)
    acc1,recall=accuracy(X, Y, theta1)
    theta2 = GradientDescent_regular(X,Y)
    acc2,recall_regular=accuracy(X,Y,theta2)

    #计算训练集的准确率和召回率
    print("训练集无正则准确率:" + str(acc1))
    print("训练集无正则召回率:"+str(recall))
    print("训练集有正则准确率:" + str(acc2))
    print("训练集有正则召回率:"+str(recall_regular))


    # maketestdata()
    si,sis=readtestdata()
    acc_test1,recall_test1 = accuracy(si, sis, theta1)
    acc_test2,recall_test2 = accuracy(si,sis,theta2)
    print("测试集无正则准确率:" + str(acc_test1))
    print("测试集无正则召回率:"+str(recall_test1))
    print("测试集有正则准确率:" + str(acc_test2))
    print("测试集有正则召回率:" + str(recall_test2))

    #画图
    drawfigure(theta1,'training_no_regular')
    drawfigure(theta2,'training_with_regular')
    drawtestfigure(theta1, si, sis,'test_no_regular')
    drawtestfigure(theta2, si, sis,'test_with_regular')