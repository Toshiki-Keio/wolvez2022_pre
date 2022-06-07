from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


TOTAL_DIMS = 40  #@param {type: "number"}
NON_ZERO_DIMS = 2  #@param {type: "number"}

if TOTAL_DIMS < NON_ZERO_DIMS:
    raise ValueError("invalid setting")

def make_sparse_poly():
    """「スパースな」多項式関数を作成"""
    # 先頭`NON_ZERO_DIMS`個だけランダムな係数の`TOTAL_DIMS`次元の配列を作成
    non_zero_coefficients = np.random.randn(NON_ZERO_DIMS)
    zero_coefficients = np.zeros(TOTAL_DIMS - NON_ZERO_DIMS)
    sparse_coefficients = np.concatenate((non_zero_coefficients, zero_coefficients))

    # 係数をランダムに並び替える
    np.random.shuffle(sparse_coefficients)
    sparse_poly = np.poly1d(sparse_coefficients)
    return sparse_poly


def poly_to_dense_true_data(poly):
    """多項式関数からデータ点を作成"""
    true_x = np.linspace(0, 1, 201)
    true_y = poly(true_x)
    return true_x, true_y


def poly_to_sparse_noisy_data(poly):
    """多項式関数から観測用のデータ点を疎に作成"""
    noisy_x = np.linspace(0, 1, 11)
    noisy_y = poly(noisy_x) + np.random.normal(0, 0.01, len(noisy_x))
    return noisy_x, noisy_y


def poly_to_dense_noisy_data(poly):
    """多項式関数から観測用のデータ点を密に作成"""
    noisy_x = np.linspace(0, 1, 51)
    noisy_y = poly(noisy_x) + np.random.normal(0, 0.01, len(noisy_x))
    return noisy_x, noisy_y


def wrap_fit(model, X, y):
    X = X.reshape(-1, 1)
    X = PolynomialFeatures(TOTAL_DIMS).fit_transform(X)
    model.fit(X, y)
    return model


def wrap_predict(model, X):
    X = X.reshape(-1, 1)
    X = PolynomialFeatures(TOTAL_DIMS).fit_transform(X)
    y = model.predict(X)
    return y

def plot_point(true_x, true_y, noisy_x, noisy_y, pred_y=None):
    """データ点をプロット"""
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(true_x, true_y, linestyle='dashed', color='green', label="true")
    if pred_y is not None:
        ax.plot(true_x, pred_y, color='blue', label="predicted")
    ax.scatter(noisy_x, noisy_y, color='green', marker='o', label="observed data")

    ax.set_title('true data & noisy data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim((0,1))
    ax.tick_params(which='both', direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.grid()
    ax.legend()
    plt.show()


def plot_coefficients(true_coef, pred_coef=None):
    """多項式関数の係数をプロット"""
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    idx, = true_coef.nonzero()
    ax.scatter(idx, true_coef[idx], c="r", marker="x", label="true")
    if pred_coef is not None:
        idx, = pred_coef.nonzero()
        ax.stem(idx, pred_coef[idx], basefmt='g-', label="predicted")

    ax.set_title('coefficients')
    ax.set_xlim((0,TOTAL_DIMS))
    ax.tick_params(which='both', direction='in',
                   bottom=True, top=True, left=True, right=True)
    ax.grid()
    ax.legend()
    plt.show()

"""## 多項式関数の生成"""

poly = make_sparse_poly()
true_x, true_y = poly_to_dense_true_data(poly)
noisy_x, noisy_y = poly_to_sparse_noisy_data(poly)
plot_point(true_x, true_y, noisy_x, noisy_y)

lasso = wrap_fit(Lasso(alpha=0.001),
                 noisy_x, noisy_y)
pred_y = wrap_predict(lasso, true_x)
plot_point(true_x, true_y, noisy_x, noisy_y, pred_y)