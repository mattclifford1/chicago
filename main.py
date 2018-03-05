import numpy as np 
import matplotlib.pyplot as plt 
import tqdm
from sklearn import mixture
from scipy.stats import multivariate_normal
import pandas

def main():
	partitions = 4
	c = [20,50,100,120,140,160,200]
	for clust in c:
		train = [0]*4
		test = [0]*4
		print('------------------------------- '+str(clust)+' -------------------------------')
		for i in range(partitions):
			X, Y, sev, X_test, Y_test, sev_test = getData(i,partitions) #currently undersampling
			sevData = gmmData(X, Y, sev)
			testData = gmmData(X_test, Y_test, sev_test)
			######normalise heatmap and GMM, then compare closeness of the two
			reduction = 1000
			heatIm = makeHeatmap(X, Y, sev,reduction,0, test=False)
			s = heatIm.shape
			heatImTest = makeHeatmap(X_test, Y_test, sev_test, reduction,s,test=True)
			#do EM
			gmm = EM(sevData,n_components=clust,i=i)
			# print('done')
			#load guassian data computed from EM - need to have run EM before to have saved file
			means = gmm.means_
			cov = gmm.covariances_
			weights = gmm.weights_
			P = plot3d(means, cov,weights, X, Y, plot = False, shape = s)
			normTrain = heatIm/np.max(heatIm)
			normTest = heatImTest/np.max(heatImTest)

			trainError = 0 
			testError = 0
			for x in range(s[0]):
				for y in range(s[1]):
					trainError += (normTrain[x,y] - P[x,y])**2   
					testError += (normTest[x,y] - P[x,y])**2
			test[i] = testError
			train[i] = trainError
		print('train error: '+ str(train))
		print('test error:  '+ str(test))

def returnHeatData(normHeatIm, coord):
	return normHeatIm[coord[0],coord[1]]

def returnMaxProb(G, gLen, coord):
	p_s = [0]*gLen 	
	for x in range(gLen):
		grid = G[x][2]
		p_s[x] = grid[coord[0],coord[1]]
		# plt.imshow(G[x][2])
		# plt.show()
	return np.max(p_s)

def plot3d(means, cov, weights, X, Y, plot, shape):
	#make list of each guassian as np.array
	G = [0]*means.shape[0]    #initialise
	for i in range(means.shape[0]):
		G[i] = grids(means[i,:], cov[i,:,:],weights[i], X, Y, shape)
	#make guassan data into format plotly takes
	# data = [0]*means.shape[0]

	P = np.zeros([G[0][2].shape[0],G[0][2].shape[1]])
	for x in range(means.shape[0]):
		# data[x] = {'x':G[x][0],'y':G[x][1],'z':G[x][2], 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=1)}
		P +=  G[x][2]

	P = P/np.max(P) #normalise
	if plot == True:
		data = [{'x':G[0][0],'y':G[0][1],'z':P, 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=10)}]
		#plot
		import plotly.graph_objs as go
		layout = go.Layout(
		    title='Gaussian Mixture Model of Chicago Crime',
		    scene = dict(
	                    xaxis = dict(
	                        title='X'),
	                    yaxis = dict(
	                        title='Y'),
	                    zaxis = dict(
	                        title='Z'),)
		)
		fig = go.Figure(data=data, layout=layout)
		py.plot(fig,filename='GMM.html')  #offline plot
		# pyonline.iplot(fig,filename='GMM') #upload to online
	return P

def grids(mean, cov,weight, X, Y, shape):  #make grids of probabilities given guassian data
	resolutionX = np.max(X) - np.min(X)
	resolutionX = shape[0]
	resolutionY = np.max(Y) - np.min(Y)
	resolutionY = shape[1]
	# resolution = 100
	Xmesh,Ymesh = np.meshgrid(np.linspace(0,np.max(X),resolutionX),np.linspace(0,np.max(Y),resolutionY))
	# Xmesh,Ymesh = np.meshgrid(np.linspace(0,112,resolution),np.linspace(0,139,resolution))
	Xmesh = np.transpose(Xmesh)
	Ymesh = np.transpose(Ymesh)

	coord = np.zeros([Xmesh.size,2])
	# count = 0 
	# for i in tqdm.tqdm(range(Xmesh.shape[0])):
	# 	for j in range(Xmesh.shape[1]):
	# 		coord[count,0] = Xmesh[i,j]
	# 		coord[count,1] = Ymesh[i,j]
	# 		count += 1
	# print('ravel')
	coord[:,0] = Xmesh.ravel()
	coord[:,1] = Ymesh.ravel()
	# print('probs')
	probs = multivariate_normal(mean,cov).pdf(coord)
	P = np.reshape(probs,[resolutionX, resolutionY])
	# Xmesh = np.flip(Xmesh,1)
	# Ymesh = np.flip(Ymesh,1)
	# P = np.flip(P,1)
	P = P*weight
	return [Xmesh, Ymesh, P]

def EM(heatData, n_components, i):   #save EM data
	
	gmm = mixture.GaussianMixture(n_components=n_components).fit(heatData)
	return gmm

def getData(it,part):   #get certain partition of the data
	if it >= part:
		print('error - partition numerator too big')

	dataCoord = np.load('dataCoord.npy')    #load coordinate data
	X = dataCoord[0,:]                      #and separate
	Y = dataCoord[1,:]
	IUCR = np.load('IUCR.npy')        #load severity

	# ##using undersampling   *************
	# l = int(len(X)/100)
	# X = X[0:l]
	# Y = Y[0:l]
	# IUCR = IUCR[0:l]
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

	#split train and test 
	lower = int((it*len(X))/part)
	upper = int(((it+1)*len(X))/part)

	X_train = np.delete(X,np.arange(lower,upper))
	Y_train = np.delete(Y,np.arange(lower,upper))
	sev_train = np.delete(sev,np.arange(lower,upper))
	X_test = X[lower:upper]
	Y_test = Y[lower:upper]
	sev_test = sev[lower:upper]

	return X_train, Y_train, sev_train, X_test, Y_test, sev_test

def makeHeatmap(X, Y, sev, red, size, test):
	X = X/red
	Y = Y/red
	#define image size
	if test == False:
		heatIm =  np.zeros([int(np.ceil((np.max(X)))), int(np.ceil((np.max(Y+1))))])
	else:
		heatIm = np.zeros([size[0]+1,size[1]+1])
	#sum severities for image locations, n.b. x axis has to be flipped to match image coordinates
	for i in range(len(X-1)):
		heatIm[int(X[i]-1), int(Y[i]-1)] += int(sev[i])t.plot(i,j,'.')
	return heatIm

def gmmData(X, Y, sev):      #increases occurancy due to severity
	data_points = int(sum(sev))
	data = np.zeros([data_points,2])
	count = 0 
	for i in range(len(X)):
		units = int(sev[i])
		for k in range(units):
			data[count,0] = X[i]
			data[count,1] = Y[i]
			count += 1
	return data

if __name__ == "__main__":
 	main()
