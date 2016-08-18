from numpy import *


#�ʱ�������ת������
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]        #1�������������֣�0������������
    return postingList,classVec

def createVocabList(dataSet):           #�ĵ������е��ʵļ���(���ظ�)
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)      #�󲢼�
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my vocabulary!" % word
    return returnVec

#���ر�Ҷ˹������ѵ������
def trainNB0(trainMatrix,trainCategory):        #trainMatrix�ĵ�����postingListͨ��setOfWords2Vec���Եõ����Ƶľ���
    numTrainDocs = len(trainMatrix)             #trainCategory����ÿƪ�ĵ�������ǩ���ɵ�����
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)      #�������ĵ�ռ���ĵ����ı���
    p0Num = ones(numWords)             #����numWords*1����������ڱ�����������ĵ��ĵ���������
    p1Num = ones(numWords)
    p0Denom = 2.0                       #��¼���з��������ĵ����ж��ٸ�����
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:         #���ĵ��Ƿ����������ĵ�
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)         #P(w|Ci),w��һ����������ʾ�������1���������ĵ�ʱ��ÿ�����ʳ��ֵĸ�������Ϊp1Vect
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

#���ر�Ҷ˹���ຯ��
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)           #��Ϊת��Ϊlog��������+��sum����ʾ����log��P(w|Ci)*P(Ci))��ֵ
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

#���ر�Ҷ˹�ʴ�ģ��
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#�ļ������������������ʼ����Ժ���
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)           ####################��Ӱ����#######
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):                                   #���벢�����ı��ļ�,ÿ���ļ����¸�ѡ25���ʼ�
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
    for i in range(10):                                     #�������ѵ����,10�����Լ�
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


