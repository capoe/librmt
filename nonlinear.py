import numpy as np
import numpy.linalg as la
from sklearn import linear_model


def gp_interpolate_1d(xs, ys, xs_target, sigma, lambda_reg):
    xs = np.array(xs)
    ys = np.array(ys)
    if type(xs_target) == int:
        xs_target = np.linspace(np.min(xs), np.max(xs), xs_target)

    ys_avg = np.average(ys)
    n_train = xs.shape[0]
    n_test = xs_target.shape[0]

    ys = ys - ys_avg

    k_train = np.tile(xs, (n_train,1))**2 + np.tile(xs, (n_train,1)).T**2 - 2*np.outer(xs, xs)
    k_train = np.exp(-0.5*k_train/sigma**2)
    k_train_reg = k_train + np.identity(n_train)*lambda_reg

    k_test = np.tile(xs, (n_test,1))**2 + np.tile(xs_target, (n_train,1)).T**2 - 2*np.outer(xs_target, xs)
    k_test = np.exp(-0.5*k_test/sigma**2)

    weights = la.inv(k_train_reg).dot(ys)
    ys_target = k_test.dot(weights)

    return xs_target, ys_target + ys_avg


def interpolate_linear_1d(xs, ys, xs_target):
    xs = np.array(xs)
    ys = np.array(ys)
    if type(xs_target) == int:
        xs_target = np.linspace(np.min(xs), np.max(xs), xs_target)
    model = linear_model.LinearRegression(fit_intercept=True, normalize=False)

    n_train = xs.shape[0]
    n_test = xs_target.shape[0]

    xs = xs.reshape((n_train, 1))
    ys = ys.reshape((n_train, 1))
    xs_target = xs_target.reshape((n_test, 1))

    model.fit(xs, ys)

    ys_target = model.predict(xs_target)
    return xs_target.flatten(), ys_target.flatten()
