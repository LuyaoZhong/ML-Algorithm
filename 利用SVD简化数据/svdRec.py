def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#相似度计算
from numpy import *
from numpy import linalg as la

def eulidSim(inA,inB):                  #欧氏距离
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB):                  #皮尔逊相关系数
    if len(inA) < 3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar = 0)[0][1]

def cosSim(inA,inB):                    #余弦相似度
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


#基于物品相似度的推荐引擎
def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):                                                  #该循环大体上是对用户评过分的每个物品进行遍历，并将它与其他物品比较
        userRating = dataMat[user,j]
        if userRating==0:
            continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if  len(overLap)==0:
            similarity=0
        else:       #计算两种物品的相似度
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])          #simMeas:计算相似度的方法，有三种，见相似度计算
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal


def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):        #指定用户user，可返回该用户对未评分物品的预测评分
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]


#基于SVD的评分估计
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])                #建立对角矩阵
    xformedItems = dataMat.T*U[:,:4]*Sig4.I     #构建转换后的物品
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating==0 or j==item:
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print "the %d and %d similarity is : %f" % (item,j,similarity)
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal

#图像压缩函数
def printMat(inMat,thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k])>thresh:
                print 1,
            else:
                print 0,
        print ''

def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print "***original matrix***"
    printMat(myMat,thresh)
    U,Sigma,VT=la.svd(myMat)
    SigRecon = mat(zeros((numSV,numSV)))
    for k in range(numSV):
        SigRecon[k,k]=Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print "***reconstruct matrix using %d singular values***" % numSV
    printMat(reconMat,thresh)












