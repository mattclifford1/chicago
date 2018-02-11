import pandas
import numpy as np
from tqdm import tqdm

#define column names
colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data

#extract useful columns to lists
X = data.XCoordinate.tolist()
Y = data.YCoordinate.tolist()
P = data.PrimaryType.tolist()
B = data.Ward.tolist()
La = data.Latitude.tolist()
Lo = data.Longitude.tolist()

#convert to numpy array for ease of use
X = np.array(X)
Y = np.array(Y)
P = np.array(P)
B = np.array(B)
La = np.array(La)
Lo = np.array(Lo)

#get rid of first column containing string of column name
X = np.delete(X,0)
Y = np.delete(Y,0)
P = np.delete(P,0)
B = np.delete(B,0)
La = np.delete(La,0)
Lo = np.delete(Lo,0)

#get rid of any incomplete 'nan' entries from the data (using indexing for speed instead of deleting)
dataLen = len(X)
dataLen2 = 0
for i in tqdm(range(len(X))):
    if X[i] == 'nan':     #n.b. no entry in X iff no entry in Y (true for our data)
        dataLen -= 1
        dataLen2 += 1

Xclean = np.zeros(dataLen)    #define arrays
Yclean = np.zeros(dataLen)
Laclean = np.zeros(dataLen)
Loclean = np.zeros(dataLen)
Pclean = [0] * dataLen

Beat = np.zeros(dataLen2)
BeatClean = np.zeros(dataLen)
P_delete = [0] * dataLen2

ind = 0
ind2 =0

for i in tqdm(range(len(X))):
    if X[i] != 'nan':
        Xclean[ind] = X[i]
        Yclean[ind] = Y[i]
        Pclean[ind] = P[i]
        Laclean[ind] = La[i]
        Loclean[ind] = Lo[i]
        BeatClean[ind] = B[i]
        ind += 1
    else:
        Beat[ind2] = B[i]
        P_delete[ind2] = P[i]
        ind2 += 1

        
        
#save arrays
dataCoord = np.array([Xclean,Yclean])
LatLong = np.array([Laclean, Loclean])

np.save('dataCoord.npy',dataCoord)
np.save('severity.npy',Pclean)
np.save('beats.npy', Beat)
np.save('beatsClean.npy', BeatClean)
np.save('severity2.npy', P_delete)

#Save latitude and longitude
np.save('latLong', LatLong)





# import scipy.io

# scipy.io.savemat('data.mat', dict(X=Xclean, Y=Yclean))
