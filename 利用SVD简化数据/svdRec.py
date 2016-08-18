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

#���ƶȼ���
from numpy import *
from numpy import linalg as la

def eulidSim(inA,inB):                  #ŷ�Ͼ���
    return 1.0/(1.0+la.norm(inA-inB))

def pearsSim(inA,inB):                  #Ƥ��ѷ���ϵ��
    if len(inA) < 3:
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar = 0)[0][1]

def cosSim(inA,inB):                    #�������ƶ�
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


#������Ʒ���ƶȵ��Ƽ�����
def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):                                                  #��ѭ���������Ƕ��û������ֵ�ÿ����Ʒ���б�������������������Ʒ�Ƚ�
        userRating = dataMat[user,j]
        if userRating==0:
            continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if  len(overLap)==0:
            similarity=0
        else:       #����������Ʒ�����ƶ�
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])          #simMeas:�������ƶȵķ����������֣������ƶȼ���
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal==0:
        return 0
    else:
        return ratSimTotal/simTotal


def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):        #ָ���û�user���ɷ��ظ��û���δ������Ʒ��Ԥ������
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems)==0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]


#����SVD�����ֹ���
def svdEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])                #�����ԽǾ���
    xformedItems = dataMat.T*U[:,:4]*Sig4.I     #����ת�������Ʒ
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

#ͼ��ѹ������
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












