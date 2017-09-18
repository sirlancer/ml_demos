__author__ = 'lancer'

import numpy as np
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
import  os

# Produce some data
def createDataSet():
    group = np.array([[1,1],[1,1],[0,0],[0,1]])
    labels = ['A','A','B','B']
    return group, labels
# KNN
def classify0(inX, dataSet, labels, k=3):
    dataSize = dataSet.shape[0]

    diffMat = np.tile(inX, (dataSize,1)) - dataSet
    sqDistance = (diffMat ** 2).sum(axis=1)
    distance = sqDistance ** 0.5

    sortedDistIndicies = distance.argsort()
    labelCount = defaultdict(int)
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        labelCount[voteLabel] += 1

    # sortedLabelCount = sorted(labelCount.items(), key=operator.itemgetter(1), reverse=True)
    # return sortedLabelCount[0]
    sortedLabelCount = sorted(labelCount.items(), key=lambda x:x[1], reverse=True)
    return sortedLabelCount[0][0]
# Transform data
def file2matrix(filename):
    f = open(filename)
    rows = f.readlines()

    rowsNum = len(rows)
    feasNum = len(rows[0].strip().split('\t')) - 1

    returnMat = np.zeros((rowsNum, feasNum))
    labels = []
    for i in range(rowsNum):
        rowList = rows[i].strip().split('\t')
        returnMat[i,:] = rowList[:feasNum]
        labels.append(int(rowList[-1]))
    return returnMat, labels
# Normalize data
def autoNorm(data):
    minVals = np.min(data, axis=0)
    maxVals = np.max(data, axis=0)
    rangeVals = maxVals - minVals

    m,n = data.shape
    normData = np.zeros_like((m, n))
    normData = data - np.tile(minVals, (m, 1))
    normData = normData / np.tile(rangeVals, (m, 1))
    return normData, rangeVals, minVals
# Testing KNN
def datingClassTest(filename):
    testRatio = 0.1

    data, labels = file2matrix(filename)
    normData, rangeVals, minVals = autoNorm(data)

    testNums = int(data.shape[0] * testRatio)

    trainData = normData[testNums:, :]
    trainLabel = labels[testNums:]
    testData = normData[:testNums, :]
    testLabels = labels[:testNums]
    error = 0
    for i in range(testNums):
        plabel = classify0(testData[i,:], trainData, trainLabel, k=4)
        if plabel != testLabels[i]:
            error += 1
    print('the total error rate is:%.2f%%' %(100*error/float(testNums)))
# Image to vector
def img2vector(filename):
    f = open(filename)
    returnVec = np.zeros((1,1024))
    for i in range(32):
        lineStr = f.readline().strip()
        lineStr = [int(j) for j in lineStr]
        returnVec[0,32*i:32*(i+1)] = lineStr
        # for j in range(32):
        #     returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec
# Testing handwriting
def handwritingClassTest(trainPath, testPath):

    trainFileList = os.listdir(trainPath)
    m = len(trainFileList)
    trainData = np.zeros((m, 1024))
    trainLabels = []
    for i in range(m):
        filename = trainFileList[i]
        trainLabels.append(int(filename.split('.')[0].split('_')[0]))
        trainData[i,:] = img2vector('%s/%s' % (trainPath, filename))

    testFileList = os.listdir(testPath)
    testNums = len(testFileList)
    # testData = np.zeros((testNums, 1024))
    # print(trainData.shape)
    error = 0
    for j in range(testNums):
        filename = testFileList[j]
        testLabel = int(filename.split('.')[0].split('_')[0])
        testData = img2vector('%s/%s' % (testPath, filename))
        pLabel = classify0(testData, trainData, trainLabels, k=3)
        if pLabel != testLabel:
            error += 1
    print('the total error:%.2f%%' % (100.0 * error/testNums))



if __name__ == '__main__':
    # data, labels = createDataSet()
    # pLabel = classify0([0,0], data, labels, k=3)
    # print(pLabel)
    # data, labels = file2matrix('../data/datingTestSet2.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(data[:,1], data[:,2], 15.0*np.array(labels), 15.0*np.array(labels))
    # plt.show()
    # normData, rangeVals, minVals = autoNorm(data)
    # print(normData[:5,:])
    # print(rangeVals)
    # print(minVals)

    datingClassTest('../data/datingTestSet2.txt')
    # resVec = img2vector('../data/digits/trainingDigits/0_0.txt')
    # print(resVec.shape)
    # print(resVec[0,0:96])
    handwritingClassTest('../data/digits/trainingDigits', '../data/digits/testDigits')


