import numpy as np 
import matplotlib.pyplot as plt 

data = np.load('dataClean.npy')

plt.plot(data[0,:],data[1,:],'x')
plt.show()