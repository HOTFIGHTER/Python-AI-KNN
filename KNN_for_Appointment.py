# coding=utf-8
import numpy as np
import operator
from os import listdir
from collections import Counter

#读取数据集
def file2matrix(filename):
   """
   Desc:
       导入训练数据
   parameters:
       filename: 数据文件路径
   return:
       数据矩阵 returnMat 和对应的类别 classLabelVector
    """
   fr = open(filename,'r')
   #获取读取的行数
   numberOfLines=len(fr.readlines())
   # 生成对应的空矩阵
   # 例如：zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
   returnMat = np.zeros((numberOfLines, 3))
   classLabelVector = []
   index = 0
   fr = open(filename, 'r')
   for line in fr.readlines():
       # str.strip([chars]) --返回已移除字符串头尾指定字符所生成的新字符串
       line = line.strip()
       # 以 '\t' 切割字符串
       listFromLine = line.split('\t')
       # 每列的属性数据并转化为float类型
       returnMat[index, :] = listFromLine[0:3]
       # 每列的类别数据，就是 label 标签数据(测试集结果页)
       classLabelVector.append(int(listFromLine[-1]))
       index += 1
   # 返回数据矩阵returnMat和对应的类别classLabelVector
   return returnMat, classLabelVector

#归一化操作
def autoNorm(dataSet):
   """
    Desc:
        归一化特征值，消除特征之间量级不同导致的影响
    parameter:
        dataSet: 数据集
    return:
        归一化后的数据集 normDataSet. ranges和minVals即最小值与范围，并没有用到

    归一化公式：
        Y = (X-Xmin)/(Xmax-Xmin)
        其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
    """
   # 计算每种属性的行最大值、行最小值、范围
   minVals = dataSet.min(0)
   maxVals = dataSet.max(0)
   # 极差
   ranges = maxVals - minVals
   normDataSet = np.zeros(np.shape(dataSet))
   #计算行数
   m = dataSet.shape[0]
   # 生成与最小值之差组成的矩阵
   normDataSet = dataSet - np.tile(minVals, (m, 1))
   # 将最小值之差除以范围组成矩阵
   normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
   return normDataSet, ranges, minVals

#计算最小距离
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离度量 度量公式为欧氏距离
    diffMat = np.tile(inX, (dataSetSize, 1))-dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 将距离排序：从小到大
    sortedDistIndicies = distances.argsort()
    # 选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
      voteIlabel = labels[sortedDistIndicies[i]]
      classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    """
    Desc:
        对约会网站的测试方法
    parameters:
        none
    return:
        错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    print('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)

#预测网站情侣的结果
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))
    ffMiles = float(input("frequent filer miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult - 1])

#入口函数
if __name__ == '__main__':
    # test1()
     datingClassTest()
