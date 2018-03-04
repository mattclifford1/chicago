import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

m0 = np.load('modelsFull1.npy')
m1 = np.load('modelsFull50200.npy')
m2 = np.load('modelsFull160.npy')
sevData = np.load('sevData.npy')
m = np.load('snowy2.npy',encoding='bytes')
print(m.shape)
models = [0]*(8+6)
models = [0]*(11)
# for i in range(11):

# models[0] = m1[0]
# models[1] = m1[1]
# models[2] = m1[2]
# models[3] = m1[3]
# models[4] = m2[0]
# models[5] = m1[4] 

# models[0] = m0[0]
# models[1] = m0[1]
# models[2] = m0[2]
# models[3] = m0[3]
# models[4] = m0[4]
# models[5] = m0[5]
# models[6] = m0[6]
# models[7] = m0[7]
# models[8] = m1[0]
# models[9] = m1[1]
# models[10] = m1[2]
# models[11] = m1[3]
# models[12] = m2[0]
# models[13] = m1[4]
models = m
n_components = [50,70,100,130,160,200]
n_components = [5,10,14,18,23,30,40,50]
n_components = [5,10,14,18,23,30,40,50,51,70,100,130,160,200]
plt.clf()
plt.plot(n_components, [m.bic(sevData) for m in models], label='BIC')
print('done BIC')
plt.plot(n_components,[m.aic(sevData) for m in models], label='AIC')
# for m in models:
# 	print(m.bic(sevData))
# 	print(m.aic(sevData))
plt.legend(loc='best')
plt.xlabel('n_components (undersampled data)')
# np.save('models.npy',models)
plt.show()