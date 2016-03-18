#coding=utf-8
import numpy as np
def zeroMean(dataMat):
    meanVal = np.mean(dataMat,axis = 0)
    newData = dataMat - meanVal
    return newData,meanVal
def pca(dataMat,n):
    newData,meanVal = zeroMean(dataMat)
    covMat = np.cov(newData,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    eigValsIndice = np.argsort(eigVals)
    n_eigValIndice = eigValsIndice[-1:-(n+1):-1]
    n_eigVect = eigVects[:,n_eigValIndice]
    lowDDataMat=newData*n_eigVect
    reconMat=(lowDDataMat*(n_eigVect.T))+meanVal
    return lowDDataMat,reconMat
def pca_mod(dataMat,percentage=0.60):
    newData,meanVal = zeroMean(dataMat)
    covMat = np.cov(newData,rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    n=percentage2n(eigVals,percentage)
    eigValsIndice = np.argsort(eigVals)
    n_eigValIndice = eigValsIndice[-1:-(n+1):-1]
    n_eigVect = eigVects[:,n_eigValIndice]
    lowDDataMat=newData*n_eigVect
    reconMat=(lowDDataMat*(n_eigVect.T))+meanVal
    return lowDDataMat,reconMat
def percentage2n(eigVals,percentage):
    sortArray = np.sort(eigVals)
    sortArray = sortArray[-1::-1]
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum >= arraySum*percentage:
            return num
if __name__=='__main__':
    x=np.array([[1,2,3],[4,5,6]])
    print '输入的数据是',x
    lowDDataMat,reconMat = pca_mod(x)
    print 'lowDDataMat 是'
    print lowDDataMat
    print lowDDataMat.shape
#    #print 'reconMat 是'
#    #print reconMat
#    y=[2,1,4,6,7,3,10,9,5,8]
#    tmp = percentage2n(y,0.5)
#    print tmp
