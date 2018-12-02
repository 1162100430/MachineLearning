from numpy import *
import matplotlib.pyplot as pt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# 读取数据
def readdata(filename):
    dataMat = []
    read = open(filename)
    for line in read.readlines():
        eachline = line.strip().split()
        floatdata = map(float, eachline)
        dataMat.append(list(floatdata))
    return mat(dataMat)



# PCA算法
def pca(data, feature_num):
    m, n = shape(data)
    # 数据中心化
    meanvalue = mean(data, axis=0)
    newdata = data - meanvalue
    # 计算协方差矩阵
    cov_mat = cov(newdata, rowvar=0)
    print("协方差：")
    print(cov_mat)
    feature_val, feature_vect = linalg.eig(mat(cov_mat))
    print("特征值:")
    print(feature_val)
    print("特征向量:")
    print(feature_vect)
    index=[]
    for i in range(feature_num):
        max=-inf
        k=-1
        for j in range(len(feature_val)):
            if feature_val[j]>max:
                max=feature_val[j]
                k=j
        index.append(k)
        feature_val[k]=-inf
    max_feature=np.zeros((n,feature_num))
    for i in range(feature_num):
        for j in range(n):
            max_feature[j,i]=feature_vect[j,index[i]]

    max_feature=mat(max_feature)
    print("新的特征值向量:")
    print(max_feature)
    lowdata = newdata * max_feature
    the_newmat = (lowdata * max_feature.T) + meanvalue
    low=data*max_feature
    # print(lowdata)
    return low,lowdata, the_newmat


# 画出图像
def draw_figure(lowdata,mydata):
    m = shape(lowdata)[0]

    pt.figure()
    ax = pt.subplot(111,projection='3d')#projection='3d'
    # ax.scatter(mydata[:,0],mydata[:,1],mydata[:,2],c='b')
    # ax.scatter(lowdata[:,0], lowdata[:,1],c='r')
    ax.scatter(lowdata[:, 0], lowdata[:, 1], lowdata[:,2],c='r')
    pt.title('after pca data')

    pt.figure()
    ax = pt.subplot(111, projection='3d')  # projection='3d'
    ax.scatter(mydata[:, 0], mydata[:, 1], mydata[:, 2], c='b')
    pt.title('mydata')

    pt.figure()
    ax = pt.subplot(111, projection='3d')  # projection='3d'
    ax.scatter(mydata[:, 0], mydata[:, 1], mydata[:, 2], c='b')
    ax.scatter(lowdata[:, 0], lowdata[:, 1], lowdata[:, 2], c='r')
    pt.title('general data')
    # for i in range(m):
    #     pt.plot(mydata[i, 0], mydata[i, 1], 'ob')
    # for i in range(m):
    #     pt.plot(lowdata[i, 0], lowdata[i, 1], 'xr', markersize=10)
    pt.show()


def main():
    #mydata
    feature_num=2
    data=readdata('data.txt')
    low,lowdata,newdata=pca(data,feature_num)
    print('提取的主成分的值:')
    print(lowdata)
    draw_figure(newdata,data)


if __name__ == '__main__':
    main()
