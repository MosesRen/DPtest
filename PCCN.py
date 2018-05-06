# -*- coding: utf-8 -*-
import numpy as np
import math 
#2017年中文论文提出的基于成分分析的差分隐私
#零均值化  
def zeroMean(dataMat):
	#均值化        
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值  
    newData=dataMat-meanVal  
    return newData,meanVal  

def laplacenoise(k,n,epsilon,p):
    #参数含义 原矩阵为k*n 投影矩阵为p*n 隐私预算为epsilon
    #生成laplace噪声
    loc = 0.
    scale = math.sqrt(k*p)/(epsilon*k)
    lens = k*n
    noise = np.empty((n,k))
    s = np.random.laplace(loc, scale, lens)
    r=0
    for i in range(n-1):
    	for j in range(k-1):
    		noise[i,j]=s[r]
    		r+=1
    return noise
  
def pca(dataMat,n):
	#主成分分析
    newData,meanVal=zeroMean(dataMat)  
    covMat=np.cov(newData,rowvar=0)    
    #求协方差矩阵,return ndarray 若rowvar非0，一列代表一个样本，为0，一行代表一个样本
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
    lowDDataMat += laplacenoise(n,len(dataMat),0.1,20)
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  
    #重构数据  
    return lowDDataMat,reconMat 