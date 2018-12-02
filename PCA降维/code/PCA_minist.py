from numpy import *
import numpy as np
from PIL import Image

# 读取数据
def readdata(filename):
    dataMat = []
    read = open(filename)
    for line in read.readlines():
        eachline = line.strip().split()
        floatdata = map(float, eachline)
        dataMat.append(list(floatdata))
    return mat(dataMat)

# 读取minist数据集
def readminist():
    arra = []
    read = open('minist.txt')
    for line in read.readlines():
        eachline = line.strip().split()
        floatdata = map(float, eachline)
        arra.append(list(floatdata))
    return array(arra)

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
    # print(lowdata)
    return the_newmat

#显示图片信息
def ministshow(num_three):
    figure = imgs(num_three, 10, 5, 28, 28, 'L')
    figure.show()


def toimg(array):
    array = array * 255
    img = Image.fromarray(array.astype(float32))
    return img


def imgs(imgs, col, row, width, height, type):
    newimg = Image.new(type, (col * width, row * height))
    length=len(imgs)
    for i in range(length):
        each_img = toimg(np.array(imgs[i]).reshape(width, width))
        newimg.paste(each_img, ((i % col) * width, (i // col) * width))
    return newimg

#信噪比计算
def snr(data,newdata):
    data=mat(data)
    newdata=mat(newdata)
    m,n=shape(data)
    sum1=0.0
    sum2=0.0
    for i in range(m):
        for j in range(n):
            sum1=sum1+np.power(data[i,j],2)

    for i in range(m):
        for j in range(n):
            sum2=sum2+np.power(data[i,j]-newdata[i,j],2)
    return 10.0*np.log10(sum1/sum2)



def main():
    num_three = readminist()
    newima = pca(array(num_three), 150)
    sn=snr(num_three,newima)
    print('信噪比:')
    print(sn)
    ministshow(num_three)
    ministshow(newima)


if __name__ == '__main__':
    main()