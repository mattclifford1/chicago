import numpy as np 
# import matplotlib.pyplot as plt 
# from matplotlib import cm
# import plotly.offline as py
# import plotly.plotly as pyonline
from joblib import Parallel, delayed
import multiprocessing
from sklearn.cluster import KMeans , MiniBatchKMeans
def main():
	X, Y, sev = getData()
	# reduction = 1000
	# heatIm, x_min, y_min = makeHeatmap(X, Y, sev, reduction)
	# heatData = heatmapData(heatIm)
	sevData = gmmData(X, Y, sev)

	
	n_components = np.arange(1, 100)
	# score = [0]*(len(Nc))
	# for i in Nc:
	# 	print(i)
	# 	kM = MiniBatchKMeans(n_clusters=i).fit(sevData)
	# 	score[i-1] = kM.inertia_       #Sum of squared distances of samples to their closest cluster center.

	num_cores = multiprocessing.cpu_count()
	# num_cores = 1
	score = Parallel(n_jobs=num_cores)(delayed(KM)(n, sevData) for n in n_components)
	print(score)
	print(n_components)
	np.save('score',score)
	np.save('n_components',n_components)
	# plt.clf()
	# plt.plot(n_components,score)
	# # plt.title('K means error with varying number of clusters')
	# plt.xlabel('Number of clusters')
	# plt.ylabel('Error')
	# plt.savefig('kerror.png')

	# do k means
	# nC = 10
	# kM = KMeans(n_clusters=nC).fit(heatData)
	# m = kM.cluster_centers_
	# labels = kM.labels_
	# means = [0]*nC
	# cov = [0]*nC
	# for i in range(nC):
	# 	occ = (labels == i).sum()
	# 	data = [0]*occ
	# 	count = 0 
	# 	for k in range(heatData.shape[0]):
	# 		if labels[k] == i:
	# 			data[count] = heatData[k,:]
	# 			count += 1
	# 	means[i-1] = np.mean(data, axis=0)
	# 	cov[i-1] = np.cov(np.array(data).T)
	# plt.clf()
	# plt.plot(m[:,1],m[:,0],'x')
	# plt.show()
	# #do EM
	# EM(heatData)

	# #load guassian data computed from EM - need to have run EM before to have saved file
	# means = np.load('means.npy')
	# cov = np.load('cov.npy')

	#make list of each guassian as np.array
	# means = np.array(means)
	# cov = np.array(cov)
	# me = np.array(means)
	# G = [0]*me.shape[0]    #initialise
	# for i in range(me.shape[0]):
	# 	G[i] = grids(means[i], cov[i], X, Y)
	# #make guassan data into format plotly takes
	# data = [0]*me.shape[0]
	# for x in range(me.shape[0]):
	# 	data[x] = {'x':G[x][0],'y':G[x][1],'z':G[x][2], 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=1)}
	# #plot
	# import plotly.graph_objs as go
	# layout = go.Layout(
	#     title='Gaussian Mixture Model of Chicago Crime - K means',
	#     scene = dict(
 #                    xaxis = dict(
 #                        title='Latitude'),
 #                    yaxis = dict(
 #                        title='Longitude'),
 #                    zaxis = dict(
 #                        title='Probability'),)
	# )
	# fig = go.Figure(data=data, layout=layout)
	# py.plot(fig,filename='kmeans10.html')  #offline plot
	# pyonline.iplot(fig,filename='kmeans') #upload to online

	#do DBscan 
def KM(k,sevData):
	kM = MiniBatchKMeans(n_clusters=i).fit(sevData)
	print(k)
     #Sum of squared distances of samples to their closest cluster center.
	return kM.inertia_
def grids(mean, cov, X, Y):  #make grids of probabilities given guassian data
	from scipy.stats import multivariate_normal
	resolution = 100
	Xmesh,Ymesh = np.meshgrid(np.linspace(np.min(X),np.max(X),resolution),np.linspace(np.min(Y),np.max(Y),resolution))
	Xmesh,Ymesh = np.meshgrid(np.linspace(0,112,resolution),np.linspace(0,139,resolution))
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
	return [Xmesh, Ymesh, P]

def EM(heatData):   #save EM data
	from sklearn import mixture
	gmm = mixture.BayesianGaussianMixture(n_components=20).fit(heatData)
	# gmm = mixture.GaussianMixture(n_components=###).fit(heatData)
	np.save('means.npy',gmm.means_)
	np.save('cov.npy',gmm.covariances_)

def getData():
	dataCoord = np.load('dataCoord2.npy')    #load coordinate data
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
		heatIm[abs(int(y_shift[i])-heatIm.shape[0]+1), int(x_shift[i])] += int(sev[i]*10)

	#plot and save image
	plt.imshow(heatIm,cmap='hot')
	plt.xticks([])
	plt.yticks([])
	plt.savefig('heatmap.png')
	print('saved')
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
