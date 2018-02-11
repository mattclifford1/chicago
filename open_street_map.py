#IMPORTANT -- THIS MUST BE RUN IN JUPYTER NOTEBOOK ONLINE IN ORDER TO DISPLAY HEATMAP

import gmaps
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.misc import imread
import gmaps.datasets
        

gmaps.configure(api_key="AIzaSyCmrMmc8SKB8SBpGs42gxUfOy14s2UbGe8") # Your Google API key

dataCoord = np.load('latLong.npy')    #load coordinate data
X = dataCoord[0,:]                      #and separate
Y = dataCoord[1,:]
sevStr = np.load('severity.npy')        #load severity

#lookup of numberal equivelant (should be done in a dict really)
primary = ['ARSON', 'ASSAULT', 'BATTERY','BURGLARY','CONCEALED CARRY LICENSE VIOLATION','CRIM SEXUAL ASSAULT','CRIMINAL DAMAGE','CRIMINAL TRESPASS','DECEPTIVE PRACTICE','GAMBLING','HOMICIDE','HUMAN TRAFFICKING','INTERFERENCE WITH PUBLIC OFFICER','INTIMIDATION','KIDNAPPING','LIQUOR LAW VIOLATION','MOTOR VEHICLE THEFT','NARCOTICS','OBSCENITY','OFFENSE INVOLVING CHILDREN','OTHER NARCOTIC VIOLATION','OTHER OFFENSE','PROSTITUTION','PUBLIC INDECENCY','PUBLIC PEACE VIOLATION', 'ROBBERY','SEX OFFENSE','STALKING','THEFT','WEAPONS VIOLATION','NON-CRIMINAL']
severity = [0.5,0.72,0.89,0.44,0.11,0.94,0.28,0.11,0.44,0.11,0.94,1.0,0.11,0.22,1.0,0.11,0.17,0.22,0.11,0.89,0.22,0.11,0.17,0.11,0.22,0.94,0.94,0.67,0.17,0.22,0]

sev = np.zeros(len(sevStr))
#assign severity str to numerical value
for i in range(len(sevStr)):
	ind = primary.index(sevStr[i])
	sev[i] = severity[ind]


dataCoord = np.transpose(dataCoord)
fig = gmaps.figure()
fig.add_layer(gmaps.heatmap_layer(dataCoord))
fig