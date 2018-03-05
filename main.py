import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import plotly.offline as py
import plotly.plotly as pyonline
from sklearn import mixture
from joblib import Parallel, delayed
import multiprocessing
import pandas
import sys

def main():
	data = False
	criterion = False
	plot = False
	if len(sys.argv) == 1:
		print('input \'plot\' or \'criterion\' as command line arguments')
	elif len(sys.argv) > 2:
		print('too many input arguments, input \'plot\' or \'criterion\' only')
	else:
		if sys.argv[1] == 'plot':
			data = True
			criterion = False
			plot = True
			print('plotting')
		if sys.argv[1] == 'criterion':
			data = True
			criterion = True
			plot = False
			print('doing criterion')
	
	if data == True:
		X, Y, sev = getData()
		sevData = gmmData(X, Y, sev)
		np.save('sevData.npy',sevData)
		print(np.max(X))
		print(np.max(Y))
	
	if criterion == True:
		sevData = np.load('sevData.npy')

		n_components = [5,10,14,18,23,30,40,50, 70,100,130,160,200]

		num_cores = multiprocessing.cpu_count()
		models = Parallel(n_jobs=num_cores)(delayed(EM)(n) for n in n_components)

		plt.clf()
		plt.plot(n_components,[m.aic(sevData) for m in models], label='AIC')
		plt.plot(n_components, [m.bic(sevData) for m in models], label='BIC')
		plt.legend(loc='best')
		plt.xlabel('Number of Components')
		plt.ylabel('Criterion')
		np.save('models.npy',models)
		plt.show()

	if plot == True:

		runBefore = True
		if runBefore == False:
			gmm = EM(n_components=150)
			np.save('means.npy',gmm.means_)
			np.save('cov.npy',gmm.covariances_)
			np.save('weights.npy',gmm.weights_)
		#load guassian data computed from EM - need to have run EM before to have saved file
		means = np.load('means.npy')
		cov = np.load('cov.npy')
		weights = np.load('weights.npy')
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
		np.save('severities.npy', P)
		contour = True
		plot3d = True  #will need to have a plotly account logged in at terminal
		if contour == True:
			plt.clf()
			font = {'weight' : 'bold','size'   : 10}
			matplotlib.rc('font', **font)
			plt.contour(G[0][0],G[0][1],P, 15, linewidths=0.5, colors='k')
			plt.contourf(G[0][0],G[0][1],P, 15, vmax=P.max(), vmin=P.min(),cmap='BuPu')
			plt.colorbar()  # draw colorbar
			plt.xlabel('X Coordinate')
			plt.ylabel('Y Coordinate')
			plt.show()
		if plot3d == True:
			data = [{'x':G[0][0],'y':G[0][1],'z':P, 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=10,thickness=40)}]
			#plot
			import plotly.graph_objs as go
			layout = go.Layout(
			    title='Gaussian Mixture Model of Chicago Crime with '+str(means.shape[0]) +  ' Components',
			    scene = dict(
		                    xaxis = dict(
		                        title='X Coordinate'),
		                    yaxis = dict(
		                        title='Y Coordinate'),
		                    zaxis = dict(
		                        title='Severity'),),
			    font=dict(size=18)
			)
			fig = go.Figure(data=data, layout=layout)
			py.plot(fig,filename='GMM3.html')  #offline plot
			# pyonline.iplot(fig,filename='GMM3') #upload to online

def grids(mean, cov, weight, X, Y):  #make grids of probabilities given guassian data
	from scipy.stats import multivariate_normal
	resolution = 100
	Xmesh,Ymesh = np.meshgrid(np.linspace(np.min(X),np.max(X),resolution),np.linspace(np.min(Y),np.max(Y),resolution))
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
	sevData = np.load('sevData.npy')
	print(str(n_components)+' EM...')
	bayesian = False
	if bayesian == True:
		gmm = mixture.BayesianGaussianMixture(
			n_components=n_components,
			tol=0.001,
			init_params='kmeans',
			weight_concentration_prior = 0.000001,
			weight_concentration_prior_type='dirichlet_process', 
			max_iter = 1000
		).fit(sevData)
	else:
		gmm = mixture.GaussianMixture(n_components=n_components).fit(sevData)
	return gmm

def getData():
	dataCoord = np.load('dataCoord.npy')    #load x y  data
	# dataCoord = np.load('dataLL.npy')    #load lat and long data
	X = dataCoord[0,:]                      #and separate
	Y = dataCoord[1,:]
	IUCR = np.load('IUCR.npy')        #load severity
	
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

def gmmData(X, Y, sev):      #increases occurancy due to severity
	data_points = int(sum(sev))
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

if __name__ == "__main__":
 	main()
