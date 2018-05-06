# -*- coding: utf-8 -*-
import numpy as np 
#2016年AAAI论文中提出的基于PCA差分隐私保护的拉普拉斯实现
#零均值化  
def zeroMean(dataMat):
	#均值化        
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值  
    newData=dataMat-meanVal  
    return newData,meanVal  

def laplacenoise(d,epsilon,n):
	#生成laplace噪声
	loc = 0.
	scale = 2*d/(n*epsilon)
	lens = int(d*(d+1)/2)
	noise = np.empty((d,d))
	s = np.random.laplace(loc, scale, lens)
	k=0
	for i in range(d):
		for j in range(i,d):
			noise[i,j]=s[k]
			k+=1
	noise = np.triu(noise)
	noise += noise.T -np.diag(noise.diagonal())
	return noise
  
def pca(dataMat,n):
	#主成分分析
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    
    #求协方差矩阵,return ndarray 若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    # covMat = covMat*(1//len(dataMat)) 
    covMat += laplacenoise(20, 0.1, len(dataMat))
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    #求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量  
    eigValIndice=np.argsort(eigVals)            
    #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    #最大的n个特征值的下标  
    n_eigVect=eigVects[:,n_eigValIndice]        
    #最大的n个特征值对应的特征向量  
    lowDDataMat=newData*n_eigVect               
    #低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  
    #重构数据  
    return lowDDataMat,reconMat 