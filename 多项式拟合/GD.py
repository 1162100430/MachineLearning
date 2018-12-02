#不带正则项的梯度下降法

import matplotlib.pyplot as pt
from numpy import *
from scipy.interpolate import spline
import numpy as np
import random

M=10    #多项式函数的阶数加1
k=160  #训练集点的个数

# 在0-2*pi的区间上生成k个点作为输入数据
X = np.linspace(0,1,k,endpoint=True)
Y = np.sin(2*np.pi*X)
test = np.sin(2*np.pi*X)
z = ones((k,M))


# 对输入数据加入gauss噪声
# 定义gauss噪声的均值和方差
mu = 0
sigma = 0.3
for i in range(X.size):
    Y[i] =Y[i]+random.gauss(mu,sigma)


#创建训练点的矩阵X
for i in range(k):
	temp=X[i]
	p=float(1)
	for j in range(M):
		z[i][j]=p
		p=p*temp



# 两种终止条件
loop_max = 100000  # 最大迭代次数(防止死循环)
epsilon = 1e-6


# 初始化权值
np.random.seed(0)
theta = np.random.randn(M)

alpha = 0.0015  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
diff = 0.
error = np.zeros(M)
count = 0  # 循环次数
finish = 0  # 终止标志 

while count < loop_max:
	count+=1

	sum_m=np.zeros(M)

	for i in range(k):
		dif = (np.dot(theta, z[i]) - Y[i]) * z[i]
		sum_m = sum_m + dif

	theta = theta-alpha*sum_m
    # 判断是否已收敛
	if np.linalg.norm(theta-error)<epsilon:
		finish=1
		break
	else:
		error=theta

	# print('loop count = %d' % count, '\tw:',theta)

# print('loop count = %d' % count, '\tw:',theta)

x=np.linspace(0,1,k,endpoint=True)
f=0
q=float(1)
for i in range(M):
	f=f+theta[i]*q
	q=q*x

ew=0.0
for i in range(k):
	ew=ew+(f[i]-Y[i])**2
eww=np.sqrt(ew/k)

#求解测试集的f函数和误差
# 在0-2*pi的区间上生成100个点作为测试数据
XX = np.linspace(0,1,100,endpoint=True)
YY = np.sin(2*np.pi*XX)
# 对测试数据加入gauss噪声，此时的噪声与前面的不同
# 定义gauss噪声的均值和方差
mu1 = 0
sigma1 = 0.1
for i in range(XX.size):
    YY[i] =YY[i]+random.gauss(mu1,sigma1)

#求解训练集的f函数和误差
xx=np.linspace(0,1,100,endpoint=True)
ff=0
qq=float(1)
for i in range(M):
	ff=ff+theta[i]*qq
	qq=qq*xx


ew1=0.0
for i in range(100):
	ew1=ew1+(ff[i]-YY[i])**2
eww1=np.sqrt(ew1/100)

#显示误差率
print("训练集误差率："+str(eww))
print("测试集误差率："+str(eww1))

# 画出这些点
xnew = np.linspace(X.min(),X.max(),300)
power_smooth = spline(X,test,xnew)

xnew1 = np.linspace(x.min(),x.max(),300)
power_smooth1 = spline(x,f,xnew1)

# 画出这些点
pt.plot(xnew,power_smooth,color="blue",label='sin(2*pi*x)')
pt.plot(xnew1,power_smooth1,color="red",label='fx')

pt.plot(X,Y,linestyle='',marker='.',label='traindata')

pt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
pt.show()