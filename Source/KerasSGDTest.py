import KerasSGD

ksgd = KerasSGD(100)

print(ksgd.X.shape, ksgd.Y.shape)

print("===================Fitting===================")
hist = ksgd.Fitting(maxsteps = 100)


print(hist)

print(np.array([hist.history['loss'], hist.history['val_loss']]))


import matplotlib.pyplot as plt

loss_list = []
step_list = []
val_loss_list = []
for step in range(0, len(hist.history['loss'])):
    loss_list.append(hist.history['loss'][step])
    val_loss_list.append(hist.history['val_loss'][step])
    step_list.append(step)
    
# Visualization
plt.figure(1)
plt.plot(step_list, loss_list, color = 'orange', label = 'loss')
plt.plot(step_list, val_loss_list, color = 'red', label = 'val_loss')
plt.xlabel('step(epoch)')
plt.ylabel('loss')
plt.legend()
plt.show()