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
	mixture = True
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
		# np.save('sevData.npy',sevData)
		# np.save('XY.npy',sevData)

	
	# if mixture == True:
		# sevData = np.load('sevData.npy')


	if plot == True:

		X, Y= grids(X, Y)
		from sklearn.neighbors import KernelDensity
		kde = KernelDensity().fit(sevData)
		P = kde.score_samples(sevData)
		print(sevData.shape)
		print(P.shape)
		P = P.reshape(X.shape)
		data = [{'x':X,'y':Y,'z':P, 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=10)}]
		#plot
		import plotly.graph_objs as go
		layout = go.Layout(
			title='KDE',
			scene = dict(
						xaxis = dict(
							title='X'),
						yaxis = dict(
							title='Y'),
						zaxis = dict(
							title='Z'),)
		)
		fig = go.Figure(data=data, layout=layout)
		py.plot(fig,filename='kde.html')  #offline plot
		# pyonline.iplot(fig,filename='GMM2') #upload to online

def grids(X, Y):  #make grids of probabilities given guassian data

	resolution = 100
	Xmesh,Ymesh = np.meshgrid(np.linspace(np.min(X),np.max(X),resolution),np.linspace(np.min(Y),np.max(Y),resolution))
	# Xmesh,Ymesh = np.meshgrid(np.linspace(0,112,resolution),np.linspace(0,139,resolution))
	Xmesh = np.transpose(Xmesh)
	Ymesh = np.transpose(Ymesh)

	Xmesh = np.flip(Xmesh,1)
	Ymesh = np.flip(Ymesh,1)

	return Xmesh, Ymesh

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
	X = dataCoord[0,:]                      #and separate
	Y = dataCoord[1,:]
	IUCR = np.load('IUCR.npy')        #load severity

	# ##using undersampling   *************
	l = int(len(X)/1000)
	X = X[0:l]
	Y = Y[0:l]
	IUCR = IUCR[0:l]

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
