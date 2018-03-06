import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
import pandas
import scipy as sc

def main():
    time = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    heatmaps = [0]*len(time)
    i=0
    for j in time:
        X, Y, sev = getData(j)
        reduction = 1000
        if len(X) > 10:
            heatIm, x_min, y_min = makeHeatmap(X, Y, sev, reduction)
            heatmaps[i] = heatIm
            #plt.title(monthname[i])
            #plt.savefig('heatmaps_transparent_' + j + '.png',bbox_inches='tight')
            #print('saved transparent ' + j)
            #plt.clf()
            i += 1
            #print(heatIm.shape)
            x1 = 1148220
            y1 = 1899677
            sevNorm = getSev(x1, y1, heatIm, reduction, x_min, y_min)
            #print('severity at (' + str(x1)+', ' + str(y1) +') is: ' + str(sevNorm))
            heatData = heatmapData(heatIm)
    heatMax = np.zeros(len(time))
    for i in range(len(time)):
        heatMax[i] = np.max(heatmaps[i])
    maxMax = np.max(heatMax)
    normHeatmaps = [0]*len(time)
    for i in range(len(time)):
        normHeatmaps[i] = heatmaps[i]/maxMax
    for i in range(len(time)):
        heatIm = normHeatmaps[i]
        ax = plt.figure(figsize=(18,24))
        img = sc.misc.imread('chicago_image.jpg')
        plt.imshow(img, zorder=0, extent=[0, 109, 138, 0])
        plt.imshow(heatIm, zorder=1, cmap='gist_stern', alpha=0.815,vmin=0, vmax=1)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=30)
        #plt.imshow(heatIm,cmap='hot')
        plt.xticks([])
        plt.yticks([])
        plt.title(time[i] + ':00')
        plt.savefig('heatmaps_transparent_time' + time[i] + '.png',bbox_inches='tight')
        print('saved transparent ')
        plt.clf()
        
    
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

	
	# plt.show()
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

def getData(primarytype):
    dataCoord = np.load('dataCoord' + primarytype + '.npy')    #load coordinate data
    X = dataCoord[0,:]                      #and separate
    Y = dataCoord[1,:]
    IUCR = np.load('IUCR' + primarytype + '.npy')        #load severity
    
    	#lookup of numberal equivelant (should be done in a dict really)
    colnames = ['IUCR_Codes','Primary_Type','Secondary_Type','Felony_Class','Maximum','Minimum','Mean_Sentence','Severity']
    data = pandas.read_csv('AllSeverityData.csv', names=colnames)  #extract data
    
    IUCR_Codes = data.IUCR_Codes.tolist()
    severity = data.Severity.tolist()
    
    IUCR_Codes.remove('IUCR_Codes')
    severity.remove('Severity')
    
    for i in range(len(severity)):
        severity[i] = float(severity[i])
    sev = np.zeros(len(IUCR))
    	#assign severity str to numerical value
    for i in range(len(IUCR)):
        if IUCR[i] != 'f':
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
    
    x_min = 1094.231
    y_min = 1813.91
    x_len = 110.886
    y_len = 137.625
            
        #shift to [0,0] to save figure space
    x_shift = X - x_min
    y_shift = Y - y_min
            
        #define image size
    heatIm =  np.zeros([int(np.ceil((y_len+1))), int(np.ceil((x_len+1)))], dtype=np.uint16)
            
        #sum severities for image locations, n.b. x axis has to be flipped to match image coordinates
    for i in range(len(X)):
        heatIm[abs(int(y_shift[i])-heatIm.shape[0]+1), int(x_shift[i])] += sev[i]*10
            
    # heatIm = heatIm/np.max(heatIm)
    #plot and save image
    #Makes if equal to <0.5 transparent           
    #heatIm = np.ma.masked_array(heatIm, heatIm < .1)    
    
    #Image background
    #plt.figure(figsize=(6,6))
    #img = sc.misc.imread('chicago_image.jpg')
    #Adjust left, right, bottom, down coordinates or something
    #plt.imshow(img, zorder=0, extent=[0, 109, 138, 0])


    #plot and save image
    #Alpha adjusts transparency
    
    #plt.imshow(heatIm, zorder=1, cmap='gist_stern', alpha=0.815)
    #plt.colorbar()
    #plt.imshow(heatIm,cmap='hot')
    #plt.xticks([])
   # plt.yticks([])
    #plt.savefig('heatmap.png')
    #print('saved')
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


