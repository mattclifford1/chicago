import numpy as np 
import matplotlib.pyplot as plt 
from scipy.misc import imread

dataCoord = np.load('dataCoord.npy')    #load coordinate data
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
    
#find min and 'width' for image space
x_min = np.min(X)
y_min = np.min(Y)
x_len = int(np.max(X) - x_min)
y_len = int(np.max(Y) - y_min)

#shift to [0,0] to save figure space
x_shift = X - x_min
y_shift = Y - y_min

data_reduce = 1000   #make image a sensible size

#define image size
heatIm =  np.zeros([int(np.ceil((y_len+1)/data_reduce)), int(np.ceil((x_len+1)/data_reduce))], dtype=np.uint16)

#sum severities for image locations, n.b. x axis has to be flipped to match image coordinates
for i in range(len(X)):
	heatIm[abs(int(y_shift[i]/data_reduce)-heatIm.shape[0]+1), int(x_shift[i]/data_reduce)] += int(sev[i]*10)

#Makes if equal to <0.5 transparent           
#heatIm = np.ma.masked_array(heatIm, heatIm < .5)    
    
#Image background
img = imread('chicago_image.jpg')
#Adjust left, right, bottom, down coordinates or something
plt.imshow(img, zorder=0, extent=[0, 109, 138, 0])



#plot and save image
#Alpha adjusts transparency
plt.imshow(heatIm, zorder=1, cmap='hot', alpha=0.8)
plt.xticks([])
plt.yticks([])
plt.savefig('heatmap.png')
print('saved')
