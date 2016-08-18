from math import log
import operator
import treePlotter

def calcShannonEnt(dataSet):         #������ũ�أ�ֻ������Լ���ռ�����й�,������ѧϰʵս��P35
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def creatDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']        #labels����˼��ÿһ�����Դ������ʲô��˼
    return dataSet,labels

#�������ݼ�
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]         #����ȥ��axis�е����ݳ�ȡ���ϵ�һ��
            reducedFeatVec.extend(featVec[axis+1:]) #��������һ������������ݳ�ȡ����
            retDataSet.append(reducedFeatVec)       #��˸��ݵ�axis�е�ֵ���͵õ�һ�����ݼ��Ļ���
    return retDataSet

#ѡ����õ����ݼ����ַ�ʽ
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)            #����ԭʼ��ũ��
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [exsample[i] for exsample in dataSet]    #����Ψһ�ķ����ǩ�б�
        #�����ݼ���ĳһ�е�ֵ��ĳһ����ֵ������featList��
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:                #����ÿ�ֻ��ֵ���Ϣ��
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
        
#�ݹ鹹��������
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():    #��¼ÿ�����ǩ���ֵ�Ƶ�ʣ�Ȼ��������򣬷��س����������ķ�������
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [exsample[-1] for exsample in dataSet]      #�������ݼ����������
    if classList.count(classList[0])==len(classList):       #�����ȫ��ͬʱֹͣ����
        return classList[0]
    if len(dataSet[0])==1:                  #dataSet������ֻʣ��һ�У�˵���Ѿ��������������������س��ִ�������
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)    #ѡ����õ����ݼ����ַ�ʽ
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])           #��labels�ĵ�bestFeat����ǩɾ������Ϊ�Ѿ����������ǩ���������ݼ�
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree    

#ʹ�þ������ķ��ຯ��
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

#ʹ��pickleģ��洢������
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)





















