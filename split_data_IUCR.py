import pandas
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 

def data_IUCR(primarytype):
    '''
    function that outputs specific X,Y-coordinates and Primary type of the crime 
    depending on the desired time
    
    Input:
    num = number of month that you want to observe(e.g. '12' for December or '04' for April)
    
    Output:
    Xtime = X-Coordinates with respect to all crimes in the specified time 
    Ytime = Y-Coordinates with respect to all crimes in the specified time
    Ptime = Primary type of crime for all crimes in specified time
    Dtime = Checking if the program worked correctly (i.e. all values of these should be '15' or '09')
    '''
    
    #define column names
    colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
    data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data

    
    #extract useful columns to lists
    X = data.XCoordinate.tolist()
    Y = data.YCoordinate.tolist()
    P = data.PrimaryType.tolist()
    D = data.Date.tolist()
    I = data.IUCR.tolist()
    
    #convert to numpy array for ease of use
    X = np.array(X)
    Y = np.array(Y)
    P = np.array(P)
    D = np.array(D)
    I = np.array(I)
    
    #get rid of first column containing string of column name
    X = np.delete(X,0)
    Y = np.delete(Y,0)
    P = np.delete(P,0)
    D = np.delete(D,0)
    I = np.delete(I,0)
    
    #get rid of any incomplete 'nan' entries from the data (using indexing for speed instead of deleting)
    dataLen = len(X)
    for i in tqdm(range(len(X))):
    	if X[i] == 'nan':     #n.b. no entry in X iff no entry in Y (true for our data)
    		dataLen -= 1
    
    Xclean = np.zeros(dataLen)    #define arrays
    Yclean = np.zeros(dataLen)
    Pclean = [0] * dataLen
    Dclean = [0] * dataLen
    Iclean = [0] * dataLen
    ind = 0 
    
    for i in tqdm(range(len(X))):
        if X[i] != 'nan':
            Xclean[ind] = X[i]
            Yclean[ind] = Y[i]
            Pclean[ind] = P[i]
            Dclean[ind] = D[i]
            Iclean[ind] = I[i]
            ind += 1
            
    
    Pclean = np.array(Pclean) #convert to array so can find specified time
    index = np.where(Pclean==primarytype) # finding specified time
    find = index[0] #find contains all index values of where specified time is
    
    XIUCR = np.zeros(len(find))    #define new arrays to store unique data
    YIUCR = np.zeros(len(find))
    PIUCR = [0] * len(find)
    DIUCR = [0] * len(find)
    IIUCR = [0] * len(find)
    
    #loop through all index values to fill new arrays with corresponding data from clean arrays
    for i in range(len(find)):
        XIUCR[i] = Xclean[find[i]]
        YIUCR[i] = Yclean[find[i]]
        PIUCR[i] = Pclean[find[i]]
        DIUCR[i] = Dclean[find[i]]
        IIUCR[i] = Iclean[find[i]]
    
    dataCoord = np.array([XIUCR,YIUCR])

    np.save('dataCoord' + primarytype + '.npy',dataCoord)
    np.save('IUCR' + primarytype + '.npy',IIUCR)
    print(primarytype)
    print(dataCoord.shape)
    
#define column names
colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data

#extract useful columns to lists
X = data.PrimaryType.tolist()

#sort alphabetically the unique values of primary type list
Y = sorted(list(set(X)))

#get rid of first column containing string of column name
Y.remove('PrimaryType')

for i in Y:
    data_IUCR(i)
