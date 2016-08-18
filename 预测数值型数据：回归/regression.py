from numpy import *

#标准回归函数和数据导入函数
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))-1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegress(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx)==0.0:        #linalg是线性代数的库，该行计算xTx行列式的值
        print "this matrix is singular,cannot do inverse"
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws
    
#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print "yhis matrix is singular,cannot do inverse"
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m =shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


#预测鲍鱼的年龄
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

#岭回归
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "this matrix is singular,cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    #一下五行进行数据标准化，使得每维特征具有相同的重要性
    yMean = mean(yMat,0)    #对yMat的列求均值
    yMat = yMat - yMean
    xMeans = mean(xMat,0)   #对xMat的每一列求均值
    xVar = var(xMat,0)      #对xMat的每一列求方差
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))   #取lambda为exp(i-10)可以看出lambda很小很很大时对结果造成的影响
        wMat[i,:] = ws.T
    return wMat
