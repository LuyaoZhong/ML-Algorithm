#coding=utf-8
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):        #构建簇质心
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ+rangeJ*random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas = distEclud,createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):          #寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):           #更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]           #同属于一个i簇中的点
            centroids[cent,:]=mean(ptsInClust,axis=0)
    return centroids,clusterAssment

def biKmeans(dataSet,k,distMeas = distEclud):       #二分K均值聚类算法
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    while (len(centList)<k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]      #同属于一个i簇中的点
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)         #将当前簇二分
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])   #第i簇已经被分割了，计算其他簇的SSE
            print "sseSplit,and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit+sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit+sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList)          #被分割的簇，一部分保持原来的簇序号，另一部分赋予新的簇序号
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
        print "the bestCentToSplit is :",bestCentToSplit
        print "the len of bestClustAss is : ",len(bestClustAss)
        #一下是更新全局的簇信息以及各数据所属簇的情况
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]                   #被分割的簇，一部分在原基础上更新，一部分作为新的簇添加到原来的记录中
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss
    print centList
    return mat(centList),clusterAssment
























            
        
