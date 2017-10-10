__author__ = 'lancer'

import numpy as np
import matplotlib.pyplot as plt

# Load data
def loadDataSet(filename, delim='\t'):

    f = open(filename)
    stringArr = [line.strip().split(delim) for line in f.readlines()]
    return np.mat(stringArr, dtype=np.float32)

# PCA
def pca(dataMat, topNfeat):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    print('covMat shape:', end=',')
    print(covMat.shape)
    eigVals, eigVects = np.linalg.eig(covMat)
    print('eigvals shape:', end=',')
    print(eigVals)
    print('eigVects shape:', end=',')
    print(eigVects.shape)
    eigvalInds = np.argsort(eigVals)
    eigvalInds = eigvalInds[:-(topNfeat+1):-1]
    redEigVects= eigVects[:,eigvalInds]
    lowDataMat = meanRemoved * redEigVects
    reconMat = (lowDataMat * redEigVects.T) + meanVals
    return lowDataMat, reconMat

if __name__ == '__main__':

    dataMat = loadDataSet('../data/testSet13.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()
 