from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = [] 
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):       #Sigmoid函数
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):      #使用梯度上升找到最佳回归系数
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() #转置
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))                   #最佳回归系数
    for k in range(maxCycles):              #http://blog.csdn.net/dongtingzhizi/article/details/15962797
        h = sigmoid(dataMatrix*weights)     #对每一组x，结果取1的概率，即为correct的概率
        error = (labelMat - h)              #对每一组x，结果取0的概率，即为error的概率
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights
    
#画出数据集和logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.xlabel('X2')
    plt.show()

#随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha=0.01
    weights = ones(n)
    for i in range(m):                      #对数据集中的每个样本，计算样本的梯度，然后更新回归系数
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha*error*dataMatrix[i]
    return weights


#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                    #alpha每次迭代都会更新
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
    

#logistic回归分类函数
def classifyVector(inX,weights):        #计算Sigmoid值，大于0.5返回1，即表示属于1类的概率更大
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():                    #读入训练数据和测试数据集，根据训练数据集得到的参数，对测试集进行预测，最后返回错误率
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500) #用了改进的随机梯度上升算法
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is : %f "%errorRate
    return errorRate

def multiTest():        #调用colicTest十次，计算错误率的平均值
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum +=colicTest()
    print "after %d iterations the average eroor rate is : %f" % (numTests,errorSum/float(numTests))

    
    











