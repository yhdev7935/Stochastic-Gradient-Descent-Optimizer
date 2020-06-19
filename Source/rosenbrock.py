import numpy as np
import random
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    
    a, b = np.float(1.0), np.float(2.0)
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def d_rosenbrock(x, y):
    
    a, b = np.float(1.0), np.float(2.0)
    dx = (-2) * (a - x) + (-4 * b) * (y - x ** 2) * x
    dy = 2 * b * (y - x ** 2)
    return dx, dy



def rosenbrock_sgd(x, y, epoch, lr = 8e-4, eps = 1.0e-6):
    history_x, history_y = [], []
    step_list = []
    
    for step in range(epoch):

        mu = 1.4
        dx, dy = d_rosenbrock(x, y)
        xx = x - lr * mu * dx
        yy = y - lr * mu * dy
        
        history_x.append(xx)
        history_y.append(yy)
        step_list.append(step)
        
        x = xx
        y = yy
        
    return history_x, history_y, step_list
    
   
x_list, y_list, step_list = rosenbrock_sgd(x = 5, y = -20,                 epoch = 10000)

xx = np.linspace(min(x_list) * 1.3, max(x_list) * 1.3, 800)
yy = np.linspace(min(y_list) * 1.3, max(y_list) * 1.3, 600)
X, Y = np.meshgrid(xx, yy)
Z = rosenbrock(x = X, y = Y)

levels=np.logspace(-1, 3, 10)
plt.contourf(X, Y, Z, alpha=0.2, levels=levels)
plt.contour(X, Y, Z, colors="gray",
            levels=[0.4, 3, 15, 50, 150, 500, 1500, 5000])
plt.plot(1, 1, 'ro', markersize=10)
plt.xlabel('$x$')
plt.ylabel('$y$')

# =============================

#for i in range(len(x_list)):
    #plt.plot(x_list[i], y_list[i], 'bo')
    
plt.plot(x_list, y_list, marker='o', markersize = 1)