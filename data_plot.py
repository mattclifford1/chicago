import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
import plotly.offline as py
def main():
	Lat, Long, sev = getData()
	scl = [0,"rgb(0,0,0)"],[1,"rgb(255, 0, 0)"]
	data = [ dict(
	    lat = Lat,
	    lon = Long,
	    text = sev,
	    marker = dict(
	        color = sev,
	        colorscale = scl,
	        reversescale = True,
	        opacity = 0.5,
	        size = 0.5,
	        colorbar = dict(
	            thickness = 10,
	            titleside = "right",
	            outlinecolor = "rgba(68, 68, 68, 0)",
	            ticks = "outside",
	            ticklen = 3,
	            showticksuffix = "last",
	            dtick = 0.1
	        ),
	    ),
	    type = 'scattergeo'
	) ]
	layout = dict(
	    geo = dict(
	        showland = True,
	        showrivers = True,
	        # landcolor = "rgb(212, 212, 212)",
	        # subunitcolor = "rgb(255, 255, 255)",
	        # countrycolor = "rgb(255, 255, 255)",
	        showlakes = True,
	        # lakecolor = "rgb(255, 255, 255)",
	        showsubunits = True,
	        showcountries = True,
	        resolution = 50,
	        lonaxis = dict(
	            showgrid = True,
	            gridwidth = 0.5,
	            range= [ -88, -87.5 ],
	            dtick = 5
	        ),
	        lataxis = dict (
	            showgrid = True,
	            gridwidth = 0.5,
	            range= [ 41.5, 42.1 ],
	            dtick = 5
	        )
	    ),
	    title = 'Crime in Chicago',
	)
	fig = { 'data':data, 'layout':layout }
	py.plot(fig, filename='scattermap.html')
	# reduction = 1000
	# heatIm, x_min, y_min = makeHeatmap(X, Y, sev, reduction)
	# print(heatIm.shape)
	# x1 = 1148220
	# y1 = 1899677
	# sevNorm = getSev(x1, y1, heatIm, reduction, x_min, y_min)
	# print('severity at (' + str(x1)+', ' + str(y1) +') is: ' + str(sevNorm))
	# heatData = heatmapData(heatIm)

	# from sklearn import mixture
	# # gmm = mixture.BayesianGaussianMixture(n_components=20).fit(heatData)
	# c =17
	# gmm = mixture.GaussianMixture(n_components=c).fit(heatData)

	# #do 3d plot
	# from matplotlib import cm
	# from mpl_toolkits.mplot3d import Axes3D
	# from scipy.stats import multivariate_normal
	# plt.clf()
	# fig=plt.figure();
	# # ax=fig.add_subplot(111,projection='3d')
	# ax=fig.add_subplot(111)

	# for i in range(c):
	# 	plot3dGauss(gmm.means_[i,:], gmm.covariances_[i,:,:], ax, X, Y)

	
	plt.show()
	# sio.savemat('np_xector.mat', {'xect':coord})
	# labels = gmm.predict(heatData)
	# # means = hmm.means
	# print('done')
	# plt.scatter(heatData[:, 1], heatData[:, 0], c=labels, s=1, cmap='viridis')
	# plt.show()
	# plot_results(heatData, gmm.predict(heatData), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')

	#do DBscan 
def plot3dGauss(mean, cov, ax, X, Y):
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
	ax.contour(Xmesh, Ymesh, P)

def getData():
	dataLL = np.load('dataLL.npy')    #load coordinate data
	Lat = dataLL[0,:]                      #and separate
	Long = dataLL[1,:]
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
	return Lat, Long, sev

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
