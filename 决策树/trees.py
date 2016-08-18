from math import log
import operator
import treePlotter

def calcShannonEnt(dataSet):         #计算香农熵，只与类别以及所占比例有关,《机器学习实战》P35
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
    labels = ['no surfacing','flippers']        #labels的意思是每一列属性代表的是什么意思
    return dataSet,labels

#划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]         #将除去第axis列的数据抽取整合到一起
            reducedFeatVec.extend(featVec[axis+1:]) #本行与上一行完成整个数据抽取功能
            retDataSet.append(reducedFeatVec)       #如此根据第axis列的值，就得到一个数据集的划分
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)            #计算原始香农熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [exsample[i] for exsample in dataSet]    #创建唯一的分类标签列表
        #将数据集的某一列的值即某一属性值保存在featList中
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:                #计算每种划分的信息熵
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
        
#递归构建决策树
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():    #记录每个类标签出现的频率，然后进行排序，返回出现最多次数的分类名称
            classCount[vote] = 0
        classCount += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [exsample[-1] for exsample in dataSet]      #保存数据集的所有类别
    if classList.count(classList[0])==len(classList):       #类别完全相同时停止划分
        return classList[0]
    if len(dataSet[0])==1:                  #dataSet的属性只剩下一列，说明已经遍历完所有特征，返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)    #选择最好的数据集划分方式
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])           #将labels的第bestFeat个标签删除，因为已经按照这个标签划分了数据集
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree    

#使用决策树的分类函数
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

#使用pickle模块存储决策树
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)





















