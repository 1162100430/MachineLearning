from numpy import *
import numpy as np
import matplotlib.pyplot as pt

#生成数据集
def makedata():
    n1=20
    n2=30
    n3=20
    n4=30
    co=0
    sigma=0.3
    x1 = np.random.multivariate_normal([1, 5], [[sigma, co], [co, sigma+0.1]], n1)
    x2 = np.random.multivariate_normal([3, 2], [[sigma+0.2, co], [co, sigma]], n2)
    x3 = np.random.multivariate_normal([5, 5], [[sigma, co], [co, sigma+0.1]], n3)
    x4 = np.random.multivariate_normal([7, 1], [[sigma+0.2, co], [co, sigma]], n4)
    x=[]
    y=[]
    length_t1=len(x1)
    length_t2=len(x2)
    length_t3 = len(x3)
    length_t4 = len(x4)
    for i in range(length_t1):
        x.append(x1[i][0])
        y.append(x1[i][1])

    for i in range(length_t2):
        x.append(x2[i][0])
        y.append(x2[i][1])

    for i in range(length_t3):
        x.append(x3[i][0])
        y.append(x3[i][1])

    for i in range(length_t4):
        x.append(x4[i][0])
        y.append(x4[i][1])
    #写回文件
    file = open('data.txt', mode='w')
    leng=len(x)
    for i in range(leng):
        file.writelines(str(x[i]))
        file.writelines('  ')
        file.writelines(str(y[i]))
        file.writelines('\n')
    file.close()

#读取数据
def readdata(fileName):
    dataMat = []
    read = open(fileName)
    for line in read.readlines():
        eachline = line.strip().split()
        floatdata = map(float, eachline)
        dataMat.append(list(floatdata))
    return dataMat

#随机生成k个中心点
def randomcenter(k,mat):
    n=shape(mat)[1]
    centers = np.zeros((k,n))
    centers=np.mat(centers)
    for i in range(n):
        minvalue = min(mat[:,i])
        maxvalue = max(mat[:,i])
        aver=float(maxvalue)-float(minvalue)
        centers[:,i]=minvalue+aver*random.rand(k,1)
    return centers

#计算两个点之间的欧几里得距离
def Euclid_dis(x,y):
    return sqrt(sum(power(x-y,2)))

#k-means算法
def kmeans(datamat,centers,k):
    m=shape(datamat)[0]
    newcenters = centers
    newm,newn=shape(newcenters)

    newdatamat = zeros((m,2))
    tempcenters = newcenters

    flag=True
    while(flag):
        flag=False
        for i in range(m):
            mindis=inf
            minindex=-1
            for j in range(k):
                dis=Euclid_dis(datamat[i,:],newcenters[j,:])
                if dis<mindis:
                    mindis=dis
                    minindex=j

            #若点不再变化，就退出循环
            if newdatamat[i,0] != minindex:
                flag=True
            newdatamat[i,:]=minindex,mindis

        #更新中心点的坐标
        for p in range(k):
            xx=0
            yy=0
            count=0
            for t in range(m):
                if newdatamat[t,0]==p:
                    count+=1
                    xx+=datamat[t,0]
                    yy+=datamat[t,1]
            if count==0:
                xaver=xx
                yaver=yy
            else:
                xaver = xx / count
                yaver = yy / count
            newcenters[p,:]=xaver,yaver
    return newcenters,newdatamat

#画出图像
def drawfigure(datamat,k,newcenters,newdata):
    m,n=datamat.shape
    colors = ['or', 'ob', 'og', 'ok','oy']
    for i in range(m):
        index=int(newdata[i,0])
        pt.plot(datamat[i,0],datamat[i,1],colors[index])
    color = ['xr', 'xb', 'xg', 'xk','xy']
    for i in range(k):
        pt.plot(newcenters[i,0],newcenters[i,1],color[i],markersize = 10)
    pt.show()

def main():
    # makedata()
    k=4
    data = readdata('data.txt')
    datamat=mat(data)
    centers = randomcenter(k,datamat)
    newcenters,newdata=kmeans(datamat,centers,k)
    newcenters=array(newcenters)
    print("centers:")
    print(newcenters)

    count=[0 for i in range(4)]
    for i in range(4):
        for j in range(shape(newdata)[0]):
            if newdata[j, 0] == i:
                count[i]+=1
    print("分类结果：")
    print(newdata[:,0])
    print("求得的每个类别的个数:")
    print(count)
    drawfigure(datamat,k,newcenters,newdata)
    # fig = pt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(x, y, s=30, c='black')
    # pt.show()

if __name__ == '__main__':
    main()