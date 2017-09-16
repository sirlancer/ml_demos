__author__ = 'lancer'

import operator
import numpy as np
from collections import  defaultdict

# Calculate Shannon entropy
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCount = defaultdict(int)
    for row in dataSet:
        currentLabel = row[-1]
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for label in labelCount.keys():
        prob = float(labelCount[label]) / numEntries
        shannonEnt -= prob * np.log2(prob)
    return shannonEnt

# Produce some data
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    feaNames = ['no surfacing', 'flippers']
    return dataSet, feaNames

# Split dataSet by value according to featureDim(feature dimension)
def splitDataSet(dataSet, featureDim, value):
    retDataSet = []
    for row in dataSet:
        if row[featureDim] == value:
            reduceRow = row[:featureDim]
            reduceRow.extend(row[featureDim+1:])
            retDataSet.append(reduceRow)
    return retDataSet
# Split dataSet according to infomation gain
def chooseBestFeatureToSplit(dataSet):
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    infoGain = 0.0
    bestFeature = -1
    bestInfoGain = 0.0
    for i in range(numFeature):
        featureValue = [ row[i] for row in dataSet ]
        feaValueSet = set(featureValue)
        newEntropy = 0.0
        for value in feaValueSet:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = float(len(subDataSet)) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
# Set label by majority vote
def majorityCnt(labelList):
    labelCount = defaultdict(int)
    for label in labelList:
        labelCount[label] += 1
    sortedLabelCount = sorted(labelCount.items(), key=operator.itemgetter(1), reversed=True)
    return sortedLabelCount[0][0]
# Create decision_tree
def createTree(dataSet, labels):
    labelList = [ row[-1] for row in dataSet ]
    # All the labels are the same
    if labelList.count(labelList[0]) == len(labelList):
        return  labelList[0]
    # Has traversaling all the features
    if len(dataSet[0]) == 1:
        return  majorityCnt(labelList)

    feaDim = chooseBestFeatureToSplit(dataSet)
    bestFeature = labels[feaDim]

    myTree = {bestFeature:{}}
    del(labels[feaDim])

    feaValues = [ row[feaDim] for row in dataSet]
    uniqueFeaValues = set(feaValues)
    for value in uniqueFeaValues:
        subLabels = labels[:]
        myTree[bestFeature][value] = createTree(splitDataSet(dataSet, feaDim, value), subLabels)

    return myTree

if __name__ == '__main__':
    data, feaNames = createDataSet()
    print('data:%s' % data)
    # shannonEnt = calcShannonEnt(data)
    # print('shannonEnt:%f' % shannonEnt)
    # print(splitDataSet(data, 0, 1))
    # print(splitDataSet(data, 0, 0))
    # bestFeatureDim = chooseBestFeatureToSplit(data)
    # print('best feature dimension:%d' % bestFeatureDim)
    myTree = createTree(data, feaNames)
    print(myTree)

