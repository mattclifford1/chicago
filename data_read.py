import pandas
import numpy as np
from tqdm import tqdm

#define column names
colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','Primary Type','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data

#extract useful columns to lists
X = data.XCoordinate.tolist()
Y = data.YCoordinate.tolist()

#convert to numpy array for ease of use
X = np.array(X)
Y = np.array(Y)

#get rid of first column containing string of column name
X = np.delete(X,0)
Y = np.delete(Y,0)

#get rid of any incomplete 'nan' entries from the data (using indexing for speed instead of deleting)
dataLen = len(X)
for i in tqdm(range(len(X))):
	if X[i] == 'nan':     #n.b. no entry in X iff no entry in Y (true for our data)
		dataLen -= 1

Xclean = np.zeros(dataLen)    #define arrays
Yclean = np.zeros(dataLen)
ind = 0 
for i in tqdm(range(len(X))):
	if X[i] != 'nan':
		Xclean[ind] = X[i]
		Yclean[ind] = Y[i]
		ind += 1

#save array
dataClean = np.array([Xclean,Yclean])
np.save('dataClean.npy',dataClean)


