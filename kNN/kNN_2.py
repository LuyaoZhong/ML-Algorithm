from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k):     #dataSet��ѵ�����ݼ����Ƚ�inX��dataSet�е����ݵľ��룬���ж�inX
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

def file2matrix(filename):      #�ļ���Ϊ���������ļ��е����ݼ�ת��Ϊ������ʽ�洢
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)       #�м���������
    returnmat = zeros((numberOfLines,3))    #��Ϊ�����ļ��������������ά��
    classLabelVector = []                   #ÿ��������ı��
    index = 0
    for line in arrayOfLines:           #��ÿһ��ǰ3�����ݱ�����returnmat�У���Ӧ�ı�Ǳ�����classLabelVector��
        line = line.strip();
        listFromLine = line.split('\t')
        returnmat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnmat,classLabelVector

def autoNorm(dataSet):            #��׼�����ݼ���ÿ�����ݵ�ÿһά��������1  
    minVals = dataSet.min(0)      #��ά����ÿһά����Сֵ��minVals��һ����ά����
    maxVals = dataSet.max(0)      #��ά����ÿһά�����ֵ��maxVals��һ����ά���� 
    ranges = maxVals - minVals    
    normDataSet = zeros(shape(dataSet))     #shape�����������ݼ��Ǽ��м���
    m = dataSet.shape[0]                    #�����ݼ�����������m
    normDataSet = dataSet - tile(minVals,(m,1))     #�˴���tile������minVals�����ظ�m�Σ������ظ�һ�Σ����γ�m�����ݣ�ÿ�ж���minVals
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():          #���ݷ������
    hoRatio = 0.9
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)  #90%��������������
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],\
                           datingLabels[numTestVecs:m],3)               #normMat[numTestVecs:m,:]����10%��Ϊѵ�����ݣ�kΪ3��ʾѡ��3����
        print "the classifier came back with: %d,the real answer is :%d"\
              %(classifierResult,datingLabels[i])
        if (classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print "the total error rate is:%f"%(errorCount/float(numTestVecs))
  
def classifyPerson():                           #��һ�������ý����ķ�ʽ��������������ֵ������Ԥ��
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of icecream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "you will probably like this person:",resultList[classifierResult-1]

def img2vector(filename):                   #���ļ��еı�ʾ��д���ֵľ�����ʽת��Ϊ����������ʽ
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():             #��д����Ԥ��
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
        


















