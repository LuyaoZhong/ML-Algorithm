from numpy import *

#PCA算法
def loadDataSet(fileName,delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0)              #协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat))      #计算特征值和特征向量
    eigValInd = argsort(eigVals)                    #选取前topNfeat大的特征值以及对应的特征向量
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects
    reconMat =  (lowDDataMat*redEigVects.T) + meanVals
    return lowDDataMat,reconMat


#将NaN替换成平均值
def replaceNanWithMean():
    dataMat = loadDataSet('secom.data',' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i])
        dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal
    return dataMat
