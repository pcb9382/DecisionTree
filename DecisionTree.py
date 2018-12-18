

"""
机器学习之决策树
姓名：pcb
日期：2018.12.16
"""

from math import log
import operator
import matplotlib.pyplot as plt
import pickle                               #利用pickle模块存储决策树


#-*-coding:utf-8-*-

"""
计算给定数据的香农熵
"""
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)                                   #计算数据中的总数
    labelCounts={}                                            #为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel=featVec[-1]                              #键值是字典的最后一列数值
        if currentLabel not in labelCounts.keys():            #如果当前的键值不存在，则扩展字典，并将当前的键值加入字典
            labelCounts[currentLabel]=0                       #将该键加入到字典中，并给值附为0
        labelCounts[currentLabel]+=1                          #将该键的值+1，最终得到每种分类的次数
    shannonEnt=0.0                                            #计算香农熵
    for key in labelCounts:                                   #得到字典中的键
        prob=float(labelCounts[key])/numEntries               #根据键得到值，并计算该分类的值占中分类数量的比例
        shannonEnt-=prob*log(prob,2)                          #计算熵-计算所有类别所有可能值包含的信息期望值
    return shannonEnt


"""
划分数据集,当我们按照某个特征划分数据集时，就需要将所有符合要求的元素提取出来
"""
def splitDataSet(dataSet,axis,value):
    """
    :param dataSet:待划分的数据集
    :param axis:   划分数据集的特征
    :param value:  需要返回特征的值
    :return:
    """
    #Python语言在函数中传递的是列表的引用，在函数内部对列表的修改，会影响到该列表对象的整个生存周期，
    #为了消除这个不良的影响需要在函数开始声明一个新列表对象
    retDataSet=[]                                             #创建新的list对象
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])           #将所得到的列表合并，元素个数相加
            retDataSet.append(reducedFeatVec)                 #将该列表作为一个元素添加到列表中，列表中的元素数量+1
    return retDataSet


"""
#遍历整个数据集，循环计算香农熵和splitDataSet()，找到最好的特征划分方式来划分数据集
#该函数实现了选取特征，划分数据集，计算得出最好的划分数据集的特征
#数据集dataSet必须满足两个要求：
   1.数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度
   2.数据的最后一列或者每个实例的最后一个元素是当前实例的标签
"""
def chooseBestFeatureToSplit(dataSet):

    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)  #计算数据集的原始香农熵，保存最初的无序度量值，用于与划分完之后的数据集计算的熵值进行比较
    bestInfoGain=0.0;bestFeature=-1

    # 遍历数据集中的所有特征，使用列表推导创建新的列表，将数据集中所有第i个特征或者所有可能存在的值写入新的list中
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]           #提取特征的每列数据
        uniqueVals=set(featList)
        newEntropy=0.0

        #遍历当前特征值中所有唯一属性值，对每个唯一属性值划分一次数据集
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy  #使用最初的原始数据集的熵值减去经过特征划分数据集的熵值，得到按照第一种特征划分的熵值差值
        if(infoGain>bestInfoGain):       #将每次按照原始数据集的熵值与特征划分的熵值之差来判断哪种特征划分的熵值最高，
            bestInfoGain=infoGain
            bestFeature=i                #比较所有特征的信息增益，返回最好特征划分的索引值
    return bestFeature

"""
多数表决：如果数据集已经处理了所有属性，但是类标签依然不唯一，此时通常会采用多数表决的方式决定改叶子结点的分类
"""
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classList.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

"""
创建树的函数代码

"""
def createTree(dataSet,labels):
    """
    :param dataSet: 数据集 ，前面提到的数据集的要求必须满足
    :param labels:  标签列表，标签列表中包含了数据集中所有特征的标签，算法本身并不需要这个变量
    :return:
    """
    classList=[example[-1] for example in dataSet] #创建classList列表变量，其中包含了数据集中的所有类标签

    #递归停止的第一个条件就是所有的类标签完全相同
    if classList.count(classList[0])==len(classList):  #统计classList中的类标签是否是classList的长度
        return classList[0]

    #递归停止的第二个条件使用完了所有特征，仍然不能将数据集换分成仅仅包含唯一类别的分组
    #采用选取次数最多的类别作为返回值
    if len(dataSet[0])==1:
        return majorityCnt(classList)

    #选取当前数据集中最好的特征变量存储在bestFeat中
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])       #删除标签列表中已经分类过的标签
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]    #复制类标签，并将其存储在新的列表变量subLabels中，使用subLabels代替原始列表
        #在每个数据集划分上递归调用函数createTree,得到的返回值插入到字典变量myTree中
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


"""
验证香农熵的计算
"""
def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]  #熵越高，表明混合的数据越多
    labels=['no surfacing','flippers']                        #我们按照获取最大信息增益的方法划分数据集
    return dataSet,labels

"""
-------------------------使用文本注解绘制树节点----------------------------------------------------------------------------
"""


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
"""
绘制带箭头的注解
"""
def plotNode(nodeTxt,centrPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords="axes fraction",xytext=centrPt,textcoords="axes fraction",\
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
"""
在父子节点中填充文本信息
"""
def plotMindText(cntrPt,parentPt,txtString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

"""
计算宽和高
"""
def plotTree(myTree,parentPt,nodeTxt):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    cntrpt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMindText(cntrpt,parentPt,nodeTxt)
    plotNode(firstStr,cntrpt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrpt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrpt,leafNode)
            plotMindText((plotTree.xOff,plotTree.yOff),cntrpt,str(key))

    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD


"""
实际绘图函数
"""

def createPlot():

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig=plt.figure(1,facecolor="white")
    fig.clf()                                                   #清空图像区
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode(u"决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode(u"叶节点",(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

"""
绘图
"""
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW;plotTree.yOff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


"""
获取树的叶节点的数目
"""
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs


"""
确定数的层数
"""
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=list(myTree.keys())[0]    #在python中dict.key()不是list类型，也不支持索引了，，解决的办法就是使用list()
    secondDict=myTree[firstStr]

    #计算遍历过程中遇到判断节点的个数，终止条件是叶子节点，一旦达到叶子结点，则从递归抵用中返回，并将树的深度+1
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':       #type()函数判断子节点是否为字典类型
            thisDepth=1+getTreeDepth(secondDict[key])    #节点是字典类型，需要递归调用getTreeDepth()
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth


"""
创建一个预先储存的树，避免每次测试代码都要重数据中创建树的麻烦
"""
def retrieveTree(i):
    listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},\
                 {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'yes'}}}}]

    return listOfTrees[i]

"""
------------------------------------------------------------------------------------------------------------------------
"""

"""
使用决策树进行分类
"""
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)        #使用indxe方法查找与列表中第一个匹配firstStr变量元素
    for key in secondDict.keys():
        if testVec[featIndex]==key:             #比较testVec变量中的值与树节点的值，如果达到叶子节点，则返回当前的节点分类标签
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else:
                classLabel=secondDict[key]

    return classLabel

"""
使用pickle模块存储决策树
"""
#将决策树写入txt文档中
def storeTree(inputTree,filename):
    fw=open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

#将决策树从txt文档读出
def grabTree(filename):
    fr=open(filename,'rb')
    return pickle.load(fr)


"""
读取隐形眼镜数据集
"""
def ReadLenses(filename):
    fr=open(filename)
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabel=['age','prescript','astigmatic','tearRate']
    return lenses,lensesLabel


def main():

# #1.----------------利用分类函数进行分类测试--------------------------
#     dataSet,labels=createDataSet()              #创建一个数据集
#     myTree = retrieveTree(0)                    #创建一个树用于测试画树的效果
#     classifyLabel=classify(myTree,labels,[1,0]) #利用决策树进行分类函数
#     print(classifyLabel)
#     storeTree(myTree,'classifierStorage.txt')   #测试利用pickle模块存储决策树
#     myTree1=grabTree('classifierStorage.txt')   #测试利用pickle模块读取决策
#------------------------------------------------------------------


#2.----------使用决策树进行预测隐形眼镜类型----------------------------
    lenses,lensesLable=ReadLenses('lenses.txt')   #加载隐形眼睛的数据集
    lensesTree=createTree(lenses,lensesLable)     #创建隐形眼镜的决策树
    print(lensesTree)                             #输出隐形眼镜决策树
    createPlot(lensesTree)                        #画出决策树的树图
#-------------------------------------------------------------------

# #3.----------局部函数测试---------------------------------------------
#     myTree=createTree(dataSet,labels)          #利用创建的数据集得到决策树
#     splitDataSet(dataSet,0,1)                  #
#     shannonEnt=calcShannonEnt(dataSet)         #测试计算香农熵的函数
#     print(myTree)
#     createPlot()
#
#     getNumLeafs(myTree)                        #测试得到叶子节点的函数
#     getTreeDepth(myTree)                       #测试得到树的深度的函数
#     createPlot(myTree)                          #测试画决策树的函数
# #-------------------------------------------------------------------

if __name__=="__main__":
    main()
