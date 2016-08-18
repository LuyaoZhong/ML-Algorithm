from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k):     #dataSet是训练数据集，比较inX与dataSet中的数据的距离，来判定inX
    dataSetSize = dataSet.shape[0]    
    diffMat = tile(inX,(dataSetSize,1))-dataSet

    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):      #文件名为参数，将文件中的数据集转换为矩阵形式存储
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)       #有几个数据项
    returnmat = zeros((numberOfLines,3))    #因为给的文件里的数据相是三维的
    classLabelVector = []                   #每个数据项的标记
    index = 0
    for line in arrayOfLines:           #将每一行前3个数据保存在returnmat中，对应的标记保存在classLabelVector中
        line = line.strip();
        listFromLine = line.split('\t')
        returnmat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnmat,classLabelVector

def autoNorm(dataSet):            #标准化数据集，每个数据的每一维都不超过1  
    minVals = dataSet.min(0)      #三维数据每一维的最小值，minVals是一个三维数组
    maxVals = dataSet.max(0)      #三维数据每一维的最大值，maxVals是一个三维数组 
    ranges = maxVals - minVals    
    normDataSet = zeros(shape(dataSet))     #shape函数返回数据集是几行几列
    m = dataSet.shape[0]                    #将数据集的行数赋给m
    normDataSet = dataSet - tile(minVals,(m,1))     #此处的tile函数将minVals按行重复m次，按列重复一次，即形成m行数据，每行都是minVals
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():          #数据分类测试
    hoRatio = 0.9
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  #90%的数据用来测试
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],\
                           datingLabels[numTestVecs:m],3)               #normMat[numTestVecs:m,:]将后10%作为训练数据，k为3表示选择3近邻
        print "the classifier came back with: %d,the real answer is :%d"\
              %(classifierResult,datingLabels[i])
        if (classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print "the total error rate is:%f"%(errorCount/float(numTestVecs))
  
def classifyPerson():                           #这一部分是用交互的方式，输入三个属性值，进行预测
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of icecream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "you will probably like this person:",resultList[classifierResult-1]

def img2vector(filename):                   #将文件中的表示手写数字的矩阵形式转换为行向量的形式
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():             #手写数字预测
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    n = len(testFileList)
    errorCount = 0.0
    for i in range(n):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        testVector = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify(testVector,trainingMat,hwLabels,3)
        print "the classifier came back with : %d,the real answer is : %d" % (classifierResult,classNumStr)
        if (classifierResult != classNumStr):
            errorCount+=1
    print "\nthe total number of errors is : %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(n))
        


















