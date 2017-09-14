from astropy.wcs.utils import custom_frame_mappings

__author__ = 'lancer'
import numpy as np
from sklearn import  datasets, linear_model
import  matplotlib.pyplot as plt

def genetate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.2)
    return X, y

def classify(X, y):
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf

def plot_decision_boundary(pred_func, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # colors = plt.get_cmap()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0],X[:,1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def visualize(X, y, pred_func):
    plot_decision_boundary(lambda x:pred_func.predict(x), X, y)
    plt.title('Logistic Regression')

def main():
    X, y = genetate_data()
    clf = classify(X, y)
    visualize(X, y, clf)

if __name__ == '__main__':
    main()