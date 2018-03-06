import pandas
import numpy as np
from tqdm import tqdm
import scipy.io

def data_data():
    '''
    a function that will output the data using panda.read_csv
    '''
    #define column names
    colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
    data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data
    
    return data
    
data = data_data()
    
    
    
def data_time(num,data):
    '''
    function that outputs specific X,Y-coordinates and Primary type of the crime 
    depending on the desired time
    
    Input:
    num = time of day in 24-hour clock that wanna observe data(e.g. '15' for 3pm or '09' for 9am)
    
    Output:
    Xtime = X-Coordinates with respect to all crimes in the specified time 
    Ytime = Y-Coordinates with respect to all crimes in the specified time
    Ptime = Primary type of crime for all crimes in specified time
    Dtime = Checking if the program worked correctly (i.e. all values of these should be '15' or '09')
    #define column names
    colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
    data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data
    '''
    
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
    Itime = [0] * len(find)
    
    #loop through all index values to fill new arrays with corresponding data from clean arrays
    for i in range(len(find)):
        Xtime[i] = Xclean[find[i]]
        Ytime[i] = Yclean[find[i]]
        Ptime[i] = Pclean[find[i]]
        Dtime[i] = Dclean[find[i]]
        Itime[i] = Iclean[find[i]]
    
    #dataCoord = np.array([Xtime,Ytime])

    #np.save('dataCoord' + num + '.npy',dataCoord)
    #np.save('IUCR' + num + '.npy',Itime)
    #vect = Dtime
    #scipy.io.savemat('time' + num + '.mat',{'vect':vect})
    return Xtime, Ytime, Ptime, Dtime

i = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
Dtimesize = np.zeros(len(i))
k = 0
for j in i:
    #data_time(j,data)
    Xtime,Ytime,Ptime,Dtime = data_time(j,data)
    Dtimesize[k] = len(Dtime)
    k += 1