import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
import plotly.offline as py
import plotly.plotly as pyonline
from sklearn import mixture
from joblib import Parallel, delayed
import multiprocessing

def main():
	data = True
	mixture = False
	plot = True

	if data == True:
		X, Y, sev = getData()
		reduction = 1000
		heatIm, x_min, y_min = makeHeatmap(X, Y, sev, reduction)
		# gmmData ,x,y = makeHeatmap(X, Y, sev, 1)
		# heatData = heatmapData(gmmData)
		Xnorm = X/np.max(X)
		Ynorm = Y/np.max(Y)
		sevData = gmmData(Xnorm, Ynorm, sev)
		sevData = gmmData(X, Y, sev)
		np.save('sevData.npy',sevData)
		# np.save('XY.npy',sevData)

	
	if mixture == True:
		sevData = np.load('sevData.npy')
		# #do EM
		

		n_components = [5,10,14,18,23,30,40,50, 70,90,100,120,140,155,170,200]
		# n_components = [150]
		
		# models = [0]*len(n_components)
		# count = 0
		# for n in n_components:
		# 	models[count] = mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(sevData)
		# 	print('done '+ str(n))
		# 	count+=1
		# num_cores = multiprocessing.cpu_count()
		# num_cores = 1
		# models = Parallel(n_jobs=num_cores)(delayed(EM)(n) for n in n_components)
		# models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(sevData) for n in n_components]
		# print(models)
		# plt.clf()
		# plt.plot(n_components, [m.bic(sevData) for m in models], label='BIC')
		# plt.plot(n_components,[m.aic(sevData) for m in models], label='AIC')
		# # for m in models:
		# # 	print(m.bic(sevData))
		# # 	print(m.aic(sevData))
		# plt.legend(loc='best')
		# plt.xlabel('Number of Components')
		# np.save('modelsFullnew.npy',models)
		# plt.show()

	# models = np.load('modelsFull50200.npy')
	# gmm = models[3]
	# gmm = EM(n_components=150)
	# np.save('means.npy',gmm.means_)
	# np.save('cov.npy',gmm.covariances_)
	# np.save('weights.npy',gmm.weights_)

	if plot == True:
		#load guassian data computed from EM - need to have run EM before to have saved file
		means = np.load('means.npy')
		cov = np.load('cov.npy')
		weights = np.load('weights.npy')
		red = False
		if red == True:
			X = X/1000
			Y = Y/1000
		print(str(means.shape[0]) + ' clusters')
		#make list of each guassian as np.array
		G = [0]*means.shape[0]    #initialise
		for i in range(means.shape[0]):
			G[i] = grids(means[i,:], cov[i,:,:],weights[i], X, Y)
		#make guassan data into format plotly takes
		# data = [0]*means.shape[0]

		P = np.zeros([G[0][2].shape[0],G[0][2].shape[1]])
		for x in range(means.shape[0]):
			# data[x] = {'x':G[x][0],'y':G[x][1],'z':G[x][2], 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=1)}
			P +=  G[x][2]

		P = P/np.max(P) #normalise
		np.save('P_grid.npy',P)
		P = np.load('P_grid.npy')
		contour = False
		if contour == False:
			import matplotlib
			# P = np.nan_to_num(P)
			plt.clf()
			font = {'family' : 'normal',
        			'weight' : 'bold',
        			'size'   : 10}

			matplotlib.rc('font', **font)

			plt.contour(G[0][0],G[0][1],P, 15, linewidths=0.5, colors='k')
			plt.contourf(G[0][0],G[0][1],P, 15, vmax=P.max(), vmin=P.min(),cmap='BuPu')
			plt.colorbar()  # draw colorbar
			plt.xlabel('X Coordinate')
			plt.ylabel('Y Coordinate')
			plt.show()
		else:
			data = [{'x':G[0][0],'y':G[0][1],'z':P, 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=10)}]
			#plot
			import plotly.graph_objs as go
			layout = go.Layout(
			    title='Gaussian Mixture Model of Chicago Crime with '+str(means.shape[0]) +  ' Components',
			    scene = dict(
		                    xaxis = dict(
		                        title='X'),
		                    yaxis = dict(
		                        title='Y'),
		                    zaxis = dict(
		                        title='Z'),)
			)
			fig = go.Figure(data=data, layout=layout)
			py.plot(fig,filename='GMMtest.html')  #offline plot
			# pyonline.iplot(fig,filename='GMM2') #upload to online

def grids(mean, cov, weight, X, Y):  #make grids of probabilities given guassian data
	from scipy.stats import multivariate_normal
	resolution = 100
	Xmesh,Ymesh = np.meshgrid(np.linspace(np.min(X),np.max(X),resolution),np.linspace(np.min(Y),np.max(Y),resolution))
	# Xmesh,Ymesh = np.meshgrid(np.linspace(0,112,resolution),np.linspace(0,139,resolution))
	Xmesh = np.transpose(Xmesh)
	Ymesh = np.transpose(Ymesh)

	coord = np.empty([Xmesh.size,2])
	count = 0 
	for i in range(Xmesh.shape[0]):
		for j in range(Xmesh.shape[1]):
			coord[count,0] = Xmesh[i,j]
			coord[count,1] = Ymesh[i,j]
			count += 1

	probs = multivariate_normal(mean,cov).pdf(coord)
	P = np.reshape(probs,[resolution, resolution])
	Xmesh = np.flip(Xmesh,1)
	Ymesh = np.flip(Ymesh,1)
	P = np.flip(P,1)
	P = P*weight
	return [Xmesh, Ymesh, P]

def EM(n_components):   #save EM data
	heatData = np.load('sevData.npy')
	print(str(n_components)+' EM...')
	# gmm = mixture.BayesianGaussianMixture(
	# 	n_components=n_components,
	# 	tol=0.001,
	# 	init_params='kmeans',
	# 	weight_concentration_prior = 0.000001,
	# 	weight_concentration_prior_type='dirichlet_process', 
	# 	max_iter = 1000
	# ).fit(heatData)
	gmm = mixture.GaussianMixture(n_components=n_components).fit(heatData)
	# np.save('meansB.npy',gmm.means_)
	# np.save('covB.npy',gmm.covariances_)
	return gmm

def getData():
	dataCoord = np.load('dataCoord.npy')    #load coordinate data
	# dataCoord = np.load('dataLL.npy')    #load coordinate data
	X = dataCoord[0,:]                      #and separate
	Y = dataCoord[1,:]
	IUCR = np.load('IUCR.npy')        #load severity

	# ##using undersampling   *************
	# l = int(len(X)/1000)
	# X = X[0:l]
	# Y = Y[0:l]
	# IUCR = IUCR[0:l]

	#lookup of numberal equivelant (should be done in a dict really)
	import pandas
	colnames = ['IUCR_Codes','Primary_Type','Secondary_Type','Felony_Class','Maximum','Minimum','Mean_Sentence','Severity_Normalised','Severity_Rounded','Severity_Scaled']
	data = pandas.read_csv('AllSeverityData2.csv', names=colnames)  #extract data

	IUCR_Codes = data.IUCR_Codes.tolist()
	severity = data.Severity_Scaled.tolist()

	IUCR_Codes.remove('IUCR_Codes')
	severity.remove('Severity_Scaled')

	for i in range(len(severity)):
		severity[i] = float(severity[i])
	sev = np.zeros(len(IUCR))
	#assign severity str to numerical value
	for i in range(len(IUCR)):
		ind = IUCR_Codes.index(IUCR[i])
		sev[i] = severity[ind]

	return X, Y, sev

def makeHeatmap(X, Y, sev, data_reduce):
	#reduce data size
	X = X/data_reduce
	Y = Y/data_reduce

	#find min and 'width' for image space
	x_min = np.min(X)
	y_min = np.min(Y)
	x_len = np.max(X) - x_min
	y_len = np.max(Y) - y_min

	#shift to [0,0] to save figure space
	x_shift = X - x_min
	y_shift = Y - y_min

	#define image size
	heatIm =  np.zeros([int(np.ceil((y_len+1))), int(np.ceil((x_len+1)))], dtype=np.uint16)

	#sum severities for image locations, n.b. x axis has to be flipped to match image coordinates
	for i in range(len(X)):
		heatIm[abs(int(y_shift[i])-heatIm.shape[0]+1), int(x_shift[i])] += int(sev[i])

	#plot and save image
	plt.imshow(heatIm,cmap='hot')
	plt.xticks([])
	plt.yticks([])
	plt.savefig('heatmap.png')
	# print('saved')
	# plt.clf()
	return heatIm, x_min, y_min

def getSev(x1, y1, heatIm, data_reduce, x_min, y_min):  #x and y are point from original scaling from chicago data set
	x1 = x1/data_reduce
	y1 = y1/data_reduce

	x1_shift = int(np.round(x1 - x_min))
	y1_shift = int(np.round(y1 - y_min))

	sev1 = heatIm[y1_shift,x1_shift]
	maxSev = np.max(heatIm)
	sevNorm = sev1/maxSev

	return sevNorm
def gmmData(X, Y, sev):      #increases occurancy due to severity
	data_points = int(sum(sev))
	# data_points = int(len(X))
	data = np.zeros([data_points,2])
	count = 0 
	for i in range(len(X)):
	# for i in range(10000):
		units = int(sev[i])
		# units = int(1)
		for k in range(units):
			data[count,0] = X[i]
			data[count,1] = Y[i]
			count += 1
	return data

def heatmapData(heatIm):    #converts pixels to crime data points of the heatmap
	data_points = sum(sum(heatIm))   #number of crime units
	heatData = np.zeros([data_points,2])
	count = 0
	for i in range(heatIm.shape[0]):
		for j in range(heatIm.shape[1]):
			units = heatIm[i,j]
			for k in range(units):
				heatData[count,0] = i
				heatData[count,1] = j
				count += 1
	return heatData

if __name__ == "__main__":
 	main()
