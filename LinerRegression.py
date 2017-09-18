import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, cross_validation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def load_data():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def test_LinerRegression(*data):
    X_train, X_test, y_train, y_test = data;
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(X_train, y_train)
    print('LinerRegression:')
    print('\tCoefficients:%s, intercept %0.2f' %(regr.coef_, regr.intercept_))
    print('\tResidual sum of squares:%0.2f' % np.mean((regr.predict(X_test) - y_test)**2))
    print('\tScore:%0.2f' %regr.score(X_test, y_test))

def test_Ridge(*data):
    X_train, X_test, y_train, y_test = data;
    regr = linear_model.Ridge(normalize=True)
    regr.fit(X_train, y_train)
    print('Ridge:')
    print('\tCoefficients:%s, intercept %.2f' %(regr.coef_, regr.intercept_))
    print('\tResidula sum of squares:%0.2f' % np.mean((regr.predict(X_test) - y_test)**2))
    print('\tScore:%.2f' % regr.score(X_test, y_test))

def test_Ridge_alpha(*data):
    X_train, X_test, y_train, y_test = data;
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha, normalize=True)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Rideg")
    plt.show()

def test_Lasso(*data):
    x_train, X_test, y_train, y_test = data;
    regr = linear_model.Lasso(normalize=True)
    regr.fit(X_train, y_train)
    print("Lasso:")
    print('\tCoefficient:%s, intercept %.2f' %(regr.coef_, regr.intercept_))
    print("\tResidual sum of squares:%.2f" % np.mean((regr.predict(X_test)-y_test)**2))
    print('Score:%.2f' %regr.score(X_test, y_test))

def test_Lasso_alpha(*data):
    X_train, X_test, y_train, y_test = data;
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Lasso(normalize=True, alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title('Lasso')
    plt.show()

def test_ElasticNet_alpha_rho(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2,2)
    rhos = np.linspace(0.01,1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=rho, normalize=True)
            regr.fit(X_train, Y_train)
            scores.append(regr.score(X_test, y_test))
    alphas, rhos = np.meshgrid(alphas, rhos)
    scores = np.array(scores).reshape(alphas.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas, rhos, scores, rstride=1, cstride=1, linewidth=0)
    # plt.colorbar(surf)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data()
    test_LinerRegression(X_train, X_test, Y_train, Y_test)
    test_Ridge(X_train, X_test, Y_train, Y_test)
    # test_Ridge_alpha(X_train, X_test, Y_train, Y_test)
    test_Lasso(X_train, X_test, Y_train, Y_test)
    # test_Lasso_alpha(X_train, X_test, Y_train, Y_test)
    test_ElasticNet_alpha_rho(X_train, X_test, Y_train, Y_test)