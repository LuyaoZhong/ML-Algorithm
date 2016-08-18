from numpy import *

#��׼�ع麯�������ݵ��뺯��
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
    if linalg.det(xTx)==0.0:        #linalg�����Դ����Ŀ⣬���м���xTx����ʽ��ֵ
        print "this matrix is singular,cannot do inverse"
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws
    
#�ֲ���Ȩ���Իع麯��
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


#Ԥ�Ⱬ�������
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

#��ع�
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
    #һ�����н������ݱ�׼����ʹ��ÿά����������ͬ����Ҫ��
    yMean = mean(yMat,0)    #��yMat�������ֵ
    yMat = yMat - yMean
    xMeans = mean(xMat,0)   #��xMat��ÿһ�����ֵ
    xVar = var(xMat,0)      #��xMat��ÿһ���󷽲�
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))   #ȡlambdaΪexp(i-10)���Կ���lambda��С�ܴܺ�ʱ�Խ����ɵ�Ӱ��
        wMat[i,:] = ws.T
    return wMat
