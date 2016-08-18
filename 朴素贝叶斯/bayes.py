from numpy import *


#词表到向量的转换函数
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]        #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):           #文档中所有单词的集合(不重复)
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)      #求并集
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my vocabulary!" % word
    return returnVec

#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):        #trainMatrix文档矩阵，postingList通过setOfWords2Vec可以得到类似的矩阵
    numTrainDocs = len(trainMatrix)             #trainCategory是由每篇文档的类别标签构成的向量
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)      #侮辱性文档占总文档数的比例
    p0Num = ones(numWords)             #构造numWords*1的零矩阵，用于保存非侮辱性文档的单词数向量
    p1Num = ones(numWords)
    p0Denom = 2.0                       #记录所有非侮辱性文档中有多少个单词
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:         #此文档是否是侮辱性文档
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)         #P(w|Ci),w是一个向量，表示当类别是1即侮辱性文档时，每个单词出现的概率向量为p1Vect
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)           #因为转换为log，所以是+和sum，表示的是log（P(w|Ci)*P(Ci))的值
    p0 = sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocatList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ["love","my","dalmation"]
    thisDoc = setOfWords2Vec(myVocabList,testEntry)
    print thisDoc,"\n",testEntry,"classifiy as : ",classifyNB(thisDoc,p0V,p1V,pAb)

#朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)           ####################会影响结果#######
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):                                   #导入并解析文本文件,每个文件夹下各选25封邮件
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):                                     #随机构建训练集,10个测试集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is :',float(errorCount)/len(testSet)


