import numpy as np 
import matplotlib.pyplot as plt 
from scipy.misc import imread
import matplotlib.cbook as cbook


dataCoord = np.load('dataCoord.npy')    #load coordinate data
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


#Image background
img = imread('chicago_image.jpg')
plt.imshow(img, zorder=0, extent=[0, 112, 138, 0])



#plot and save image
plt.imshow(heatIm, zorder=1, cmap='hot', alpha=0.8)
plt.xticks([])
plt.yticks([])
plt.savefig('heatmap.png')
print('saved')
