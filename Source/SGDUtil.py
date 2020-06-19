import numpy as np
import random # np.random

class SGDUtil:
    
    def __init__(self):
        pass
    
    @staticmethod
    def mse(w, x, y):
        t = (w[0] * x + w[1])
        return np.mean((y - t) ** 2) / (2 * len(x))
    
    @staticmethod
    def hill(x, y, t_w0, t_w1, g):
        w0 = np.linspace(t_w0 - g, t_w0 + g, 100)
        w1 = np.linspace(t_w1 - g, t_w1 + g, 100)
        x, y = np.array(x), np.array(y)

        J = np.zeros(shape = (len(w0), len(w1)))
        for i0 in range(len(w0)):
            for i1 in range(len(w1)):
                J[i0, i1] = SGDUtil.mse([w0[i0], w1[i1]], x, y)

        w0, w1 = np.meshgrid(w0, w1)

        return [w0, w1], J
    
    @staticmethod
    def make_regression(X_n):
        X = -3 + 13 * np.random.rand(X_n)
        Prm_c = [170, 108, 0.2] # parameter
        Y = Prm_c[0] - Prm_c[1] * np.exp(-Prm_c[2] * X)         + 4 * np.random.randn(X_n)
        X, Y = np.array(X), np.array(Y)
        return X, Y
    
    @staticmethod
    def shuffle(x, y):
        seed = random.random()
        random.seed(seed)
        random.shuffle(x)
        random.seed(seed)
        random.shuffle(y)