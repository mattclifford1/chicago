import pandas
import numpy as np
from tqdm import tqdm


def data_time(num):
    '''
    function that outputs specific X,Y-coordinates and Primary type of the crime 
    depending on the desired time
    
    Input:
    num = time of day in 24-hour clock that wanna observe data(e.g. '15' for 3pm)
    
    Output:
    Xtime = X-Coordinates with respect to all crimes in the specified time 
    Ytime = Y-Coordinates with respect to all crimes in the specified time
    Ptime = Primary type of crime for all crimes in specified time
    Dtime = Checking if the program worked correctly (i.e. all values of these should be '15')
    '''
    #define column names
    if len(num) == 1:
        num = '0' + num
    colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
    data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data
    
    #extract useful columns to lists
    X = data.XCoordinate.tolist()
    Y = data.YCoordinate.tolist()
    P = data.PrimaryType.tolist()
    D = data.Date.tolist()
    
    #convert to numpy array for ease of use
    X = np.array(X)
    Y = np.array(Y)
    P = np.array(P)
    D = np.array(D)
    
    #get rid of first column containing string of column name
    X = np.delete(X,0)
    Y = np.delete(Y,0)
    P = np.delete(P,0)
    D = np.delete(D,0)
    
    #get rid of any incomplete 'nan' entries from the data (using indexing for speed instead of deleting)
    dataLen = len(X)
    for i in tqdm(range(len(X))):
    	if X[i] == 'nan':     #n.b. no entry in X iff no entry in Y (true for our data)
    		dataLen -= 1
    
    Xclean = np.zeros(dataLen)    #define arrays
    Yclean = np.zeros(dataLen)
    Pclean = [0] * dataLen
    Dclean = [0] * dataLen
    ind = 0 
    
    for i in tqdm(range(len(X))):
        if X[i] != 'nan':
            Xclean[ind] = X[i]
            Yclean[ind] = Y[i]
            Pclean[ind] = P[i]
            Dclean[ind] = D[i]
            ind += 1
            
    #extract only the hour from Date column       
    for i in range(len(Dclean)):
        Dclean[i] = Dclean[i][11:13]
    
    Dclean = np.array(Dclean) #convert to array so can find specified time
    index = np.where(Dclean==num) # finding specified time
    find = index[0] #find contains all index values of where specified time is
    
    Xtime = np.zeros(len(find))    #define new arrays to store unique data
    Ytime = np.zeros(len(find))
    Ptime = [0] * len(find)
    Dtime = [0] * len(find)
    
    #loop through all index values to fill new arrays with corresponding data from clean arrays
    for i in range(len(find)):
        Xtime[i] = Xclean[find[i]]
        Ytime[i] = Yclean[find[i]]
        Ptime[i] = Pclean[find[i]]
        Dtime[i] = Dclean[find[i]]
    
    return Xtime,Ytime,Ptime,Dtime


       
