import pandas
import numpy as np
from tqdm import tqdm

#define column names
colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data

#extract useful columns to lists
X = data.XCoordinate.tolist()
Y = data.YCoordinate.tolist()
P = data.IUCR.tolist()

#convert to numpy array for ease of use
X = np.array(X)
Y = np.array(Y)
P = np.array(P)

#get rid of first column containing string of column name
X = np.delete(X,0)
Y = np.delete(Y,0)
P = np.delete(P,0)

#get rid of any incomplete 'nan' entries from the data (using indexing for speed instead of deleting)
dataLen = len(X)
for i in tqdm(range(len(X))):
	if X[i] == 'nan':     #n.b. no entry in X iff no entry in Y (true for our data)
		dataLen -= 1

Xclean = np.zeros(dataLen)    #define arrays
Yclean = np.zeros(dataLen)
Pclean = [0] * dataLen
ind = 0 
for i in tqdm(range(len(X))):
	if X[i] != 'nan':
		Xclean[ind] = X[i]
		Yclean[ind] = Y[i]
		Pclean[ind] = P[i]
		ind += 1


#save arrays
dataCoord = np.array([Xclean,Yclean])

np.save('dataCoord.npy',dataCoord)
np.save('IUCR.npy',Pclean)


# import scipy.io

# scipy.io.savemat('data.mat', dict(X=Xclean, Y=Yclean))

