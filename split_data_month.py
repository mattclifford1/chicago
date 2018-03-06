import pandas
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 

def data_month(num):
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
            
    
    #extract only the hour from Date column       
    for i in range(len(Dclean)):
        Dclean[i] = Dclean[i][0:2]
    
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

    #np.save('dataCoordmonth' + num + '.npy',dataCoord)
    #np.save('IUCRmonth' + num + '.npy',Itime)
    return Xtime, Ytime, Ptime, Dtime

i = ['01','02','03','04','05','06','07','08','09','10','11','12']
Dmonthsize = np.zeros(len(i))
k = 0
for j in i:
    #data_month(j)
    Xtime,Ytime,Ptime,Dtime = data_month(j)
    Dmonthsize[k] = len(Dtime)
    k += 1
       
#plt.plot(Dmonthsize,'.')
