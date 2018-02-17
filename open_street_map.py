#IMPORTANT -- THIS MUST BE RUN IN JUPYTER NOTEBOOK ONLINE IN ORDER TO DISPLAY HEATMAP
#THIS IS SINCE IT REQUIRES JUPYTER WIDGETS

import gmaps
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.misc import imread
import gmaps.datasets
        

gmaps.configure(api_key="AIzaSyCmrMmc8SKB8SBpGs42gxUfOy14s2UbGe8") # Your Google API key

dataCoord = np.load('latLong.npy')    #load coordinate data
X = dataCoord[0,:]                      #and separate
Y = dataCoord[1,:]

IUCR = np.load('IUCR.npy')        #load severity
#lookup of numberal equivelant (should be done in a dict really)
import pandas
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
	ind = IUCR_Codes.index(IUCR[i])
	sev[i] = severity[ind]
    
    
    
dataCoord = np.transpose(dataCoord)
fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(dataCoord))
fig