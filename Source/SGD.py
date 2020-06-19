import numpy as np
import SGDUtil            # deactivate in jupyter notebook

class SGD:
    
    def __init__(self, x, y):
        self.x = (x - min(x)) / (max(x) - min(x))
        self.y = (y - min(y)) / (max(y) - min(y))
        pass
        
    def process(self, epoch, w, lr = 0.2):
        x, y = self.x, self.y
        step, w_list, loss_list, step_list = 1, [], [], []
        
        while(step <= epoch):
            SGDUtil.shuffle(x, y)
            loss = 0
            dw0, dw1 = 0.0, 0.0
            
            for i in range(len(x)):
                pre_y = w[0] * x[i] + w[1]
                loss = ((y[i] - pre_y) ** 2) / 2
                
                dw0 = (pre_y - y[i]) * x[i]
                dw1 = (pre_y - y[i])
                
                w[0] = w[0] - lr * dw0
                w[1] = w[1] - lr * dw1
                
                w_list.append([w[0], w[1]])
                loss_list.append(loss)
                step_list.append(step)
                
                step += 1
                
                if(step > epoch):
                    return {'w': w_list, 'loss': loss_list, 'step': step_list}