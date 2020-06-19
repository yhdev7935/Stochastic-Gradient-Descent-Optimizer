import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import SGDUtil
import SGD

x, y = SGDUtil.make_regression(30)

plt.figure(1)
plt.plot(x, y, marker='o', linestyle='None')
# linestyle='None' 선 제거
plt.show()

sgd = SGD(x, y)
dict_result = sgd.process(epoch = 250, 
                          w = np.array([30., -40.]), lr = 0.2)
                          
                          
w_list = dict_result['w']
loss_list = dict_result['loss']
step_list = dict_result['step']

plt.figure(1)
plt.xlabel('step(epoch)')
plt.ylabel('loss')
plt.plot(step_list, loss_list, color = 'orange')
plt.show()

t_w0, t_w1 = w_list[len(w_list) - 1]
t_loss = loss_list[len(loss_list) - 1]
print("t_w0: ", t_w0, "t_w1: ", t_w1, "t_loss: ", t_loss)

hW, hJ = SGDUtil.hill(x, y, t_w0, t_w1, 50)
plt.figure(1)
plt.contourf(hW[0], hW[1], hJ, 15, alpha = 0.5, cmap = plt.cm.hot)
C = plt.contour(hW[0], hW[1], hJ, 15, alpha = 0.5, colors='black')
plt.clabel(C,inline=True)
plt.xlabel('$w_0$')
plt.ylabel('$w_1$')
plt.show()