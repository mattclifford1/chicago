import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
import plotly.offline as py
import plotly.plotly as pyonline
def main():
	X, Y, sev = getData()
	# reduction = 1000
	# heatIm, x_min, y_min = makeHeatmap(X, Y, sev, reduction)
	# heatData = heatmapData(heatIm)

	#do EM
	# EM(heatData)

	#load guassian data computed from EM - need to have run EM before to have saved file
	means = np.load('means.npy')
	cov = np.load('cov.npy')

	#make list of each guassian as np.array
	G = [0]*means.shape[0]    #initialise
	for i in range(means.shape[0]):
		G[i] = grids(means[i,:], cov[i,:,:], X, Y)
	#make guassan data into format plotly takes
	data = [0]*means.shape[0]
	for x in range(means.shape[0]):
		data[x] = {'x':G[x][0],'y':G[x][1],'z':G[x][2], 'type':'surface','text':dict(a=3),'colorscale':'Jet','colorbar':dict(lenmode='fraction', nticks=1)}
	#plot
	import plotly.graph_objs as go
	layout = go.Layout(
	    title='Gaussian Mixture Model of Chicago Crime',
	    scene = dict(
                    xaxis = dict(
                        title='Latitude'),
                    yaxis = dict(
                        title='Longitude'),
                    zaxis = dict(
                        title='Probability'),)
	)
	fig = go.Figure(data=data, layout=layout)
	py.plot(fig,filename='GMM.html')  #offline plot
	pyonline.iplot(fig,filename='GMM') #upload to online

	#do DBscan 

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
	dataCoord = np.load('dataCoord.npy')    #load coordinate data
	X = dataCoord[0,:]                      #and separate
	Y = dataCoord[1,:]
	sevStr = np.load('severity.npy')        #load severity

	#lookup of numberal equivelant (should be done in a dict really)
	primary = ['ARSON', 'ASSAULT', 'BATTERY','BURGLARY','CONCEALED CARRY LICENSE VIOLATION','CRIM SEXUAL ASSAULT','CRIMINAL DAMAGE','CRIMINAL TRESPASS','DECEPTIVE PRACTICE','GAMBLING','HOMICIDE','HUMAN TRAFFICKING','INTERFERENCE WITH PUBLIC OFFICER','INTIMIDATION','KIDNAPPING','LIQUOR LAW VIOLATION','MOTOR VEHICLE THEFT','NARCOTICS','OBSCENITY','OFFENSE INVOLVING CHILDREN','OTHER NARCOTIC VIOLATION','OTHER OFFENSE','PROSTITUTION','PUBLIC INDECENCY','PUBLIC PEACE VIOLATION', 'ROBBERY','SEX OFFENSE','STALKING','THEFT','WEAPONS VIOLATION','NON-CRIMINAL']
	severity = [0.5,0.72,0.89,0.44,0.11,0.94,0.28,0.11,0.44,0.11,0.94,1.0,0.11,0.22,1.0,0.11,0.17,0.22,0.11,0.89,0.22,0.11,0.17,0.11,0.22,0.94,0.94,0.67,0.17,0.22,0]

	sev = np.zeros(len(sevStr))
	#assign severity str to numerical value
	for i in range(len(sevStr)):
		ind = primary.index(sevStr[i])
		sev[i] = severity[ind]
	sev = sev/(np.max(sev)) #normalise
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
