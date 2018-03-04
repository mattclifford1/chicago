import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

m0 = np.load('modelsFull1.npy')
m1 = np.load('modelsFull50200.npy')
m2 = np.load('modelsFull160.npy')
sevData = np.load('sevData.npy')
m = np.load('snowyModels.npy',encoding='bytes')
print(m)
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
n_components = [5,10,14,18,23,30,40,50,70,90,100]
# [110,130,140,150,160]
a=[134685123.934457, 134064113.95021662, 133657982.56029885, 133517234.68430117, 133303669.48480266, 133104187.67257656, 132936219.62107387, 132857318.67606586, 132665544.07978442, 132544532.6185391, 132509434.5372207]
b=[134685498.4310053, 134064875.8569873, 133659054.39524743, 133518616.44742765, 133305438.65815152, 133106499.22023675, 132939305.98917882, 132861179.86461557, 132670954.90922365, 132551493.08886783, 132517169.8279942]
plt.plot(n_components,a,label='AIC')
plt.plot(n_components,b,label='BIC')
plt.show()
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