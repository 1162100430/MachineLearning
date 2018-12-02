from numpy import *
import numpy as np
import matplotlib.pyplot as pt


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
    newdatamat = zeros((m,2))
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


#计算高斯概率密度，x是数据，miu是均值数组，sigma是协方差矩阵
def gaussian(x,miu,sigma):
    m=np.shape(sigma)[0]
    sigmadet=np.linalg.det(sigma + np.eye(m) * 0.001)
    sigmainv=np.linalg.inv(sigma+np.eye(m)*0.001)
    x_miu=(x-miu).reshape((1,m))
    #计算高斯概率密度
    p=1.0/(np.power(np.power(np.pi*2,m)*np.abs(sigmadet),0.5))*\
      np.exp(-0.5*x_miu.dot(sigmainv).dot(x_miu.T))[0][0]
    return p

def initdata(k,datamat):
    centers = randomcenter(k, mat(datamat))
    newcenters, newdata = kmeans(mat(datamat), centers, k)
    len = shape(newdata)[0]
    newcenters=array(newcenters)
    count=[0 for i in range(k)]
    pi=np.zeros(k)
    for i in range(len):
        for j in range(k):
            if int(newdata[i, 0]) == j:
                count[j]+=1
    sum=np.sum(count)
    for i in range(k):
        pi[i]=1.0*count[i]/sum

    m,n=np.shape(datamat)
    # pi=np.ones(k)
    # pi=pi/sum(pi)
    sigma=[]
    miu=newcenters
    for i in range(k):
        sig=np.ones((n,n))
        sig=sig/10.0
        sigma.append(sig)
    return pi,miu,sigma

def GMM(k,datamat):
    pi,miu,sigma=initdata(k,datamat)
    m,n=np.shape(datamat)
    result_pro=[]
    result_type=[]
    func=0
    func_used=1
    esil=1e-6
    pro=[np.zeros(k) for i in range(m)]
    while True:
        func_used = func
        #E步
        for i in range(m):
            gaussvalues=[]
            for k_min in range(k):
                gaussvalue = pi[k_min]*gaussian(datamat[i], miu[k_min], sigma[k_min])
                gaussvalues.append(gaussvalue)
            gaussvalues=array(gaussvalues)
            mysum=np.sum(gaussvalues)
            if mysum!=0:
                pro[i] = gaussvalues / mysum  # 归一化
            else:
                pro[i]=gaussvalues
        #M步
        for kk in range(k):
            Nk=np.sum([pro[i][kk] for i in range(m)])
            #更新先验概率大小
            pi[kk]=1.0*Nk/m
            #更新均值
            miu[kk]=(1.0/Nk)*np.sum([pro[i][kk]*datamat[i] for i in range(m)],axis=0)
            x_miu=datamat-miu[kk]
            #更新协方差
            sigma[kk]=(1.0/Nk)*np.sum([pro[i][kk]*x_miu[i].reshape((n,1)).dot(x_miu[i].reshape((1,n))) for i in range(m)],axis=0)

        func=[]
        #计算极大似然函数
        for i in range(m):
            mid=[np.sum(pi[kk]*gaussian(datamat[i],miu[kk],sigma[kk])) for kk in range(k)]
            for i in range(k):
                if abs(mid[i]-0)<1e-6:
                    mid[i]=mid[i]
                else:
                    mid[i]=np.log(mid[i])
            func.append(mid)
        func=np.sum(func)
        #归一化
        for i in range(m):
            if abs(np.sum(pro[i])-0.0) < 1e-6:
                pro[i] = pro[i]
            else:
                pro[i] = pro[i] / np.sum(pro[i])
        result_pro=pro
        result_type=[np.argmax(pro[i]) for i in range(m)]
        print(func)
        if np.abs(func_used - func) < esil:
            break
    return result_type,result_pro,miu,sigma






def main():
    k=3
    data = readdata('uci.txt')
    m=shape(data)[0]
    datamat=array(data)
    print("似然值变化:")
    result,pro,miu,sigma=GMM(k,datamat)
    print("求得的中心点：")
    print(miu)

    count=[0 for i in range(3)]
    for i in range(3):
        for j in range(len(result)):
            if result[j] == i:
                count[i]+=1
    print("分类结果：")
    print(result)
    print("求得的每个类别的个数:")
    print(count)


    colors = ['or', 'ob', 'og', 'ok', 'oy']
    color = ['xr', 'xb', 'xg', 'xk', 'xy']
    for i in range(m):
        index=int(result[i])
        pt.plot(datamat[i,0],datamat[i,1],colors[index])
    for i in range(k):
        pt.plot(miu[i,0],miu[i,1],color[i],markersize=10)
    pt.show()


if __name__ == '__main__':
    main()