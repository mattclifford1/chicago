import GPy

GPy.plotting.change_plotting_library('plotly')
import numpy as np

from IPython.display import display

X = np.load('XYfull.npy')
Y = np.load('sev.npy')
Y = np.reshape(Y,[-1,1])

res = 1000
rand = np.random.choice(range(len(Y)), res, replace=False)
x = np.zeros([res,2])
y = np.zeros([res,1])
count = 0 
for i in rand:
	x[count,:] = X[i,:]
	y[count,0] = Y[i,0]
	count += 1
# # sample inputs and outputs
# X = np.random.uniform(-3.,3.,(50,2))
# Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05
print(y.shape)
print(x.shape)
# define kernel
ker = GPy.kern.RBF(input_dim=2, ARD=True,lengthscale=200000)# + GPy.kern.RBF(input_dim=2, ARD=True,lengthscale=0.001) #+ GPy.kern.Bias(2)
# create simple GP model
m = GPy.models.GPRegression(x,y,ker)

# optimize and plot
m.optimize(messages=True,max_f_eval = 1000)
fig = m.plot(projection='3d')
display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))
display(m)