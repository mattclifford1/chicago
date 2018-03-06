import numpy as np 
import matplotlib.pyplot as plt 


def main():
    months = ['01','02','03','04','05','06','07','08','09','10','11','12']
    monthname = ['January','February','March','April','May','June','July','August','September','October','November','December']
    j = 0
    for i in months:
        X, Y, sev = getData(i)
        reduction = 1000
        if len(X) > 1:
            heatIm, x_min, y_min = makeHeatmap(X, Y, sev, reduction)
            plt.title(monthname[j])
            plt.savefig('heatmaps_month_' + i + '.png')
            print('saved at month ' + i)
            plt.clf()
            j += 1
        else:
            print('empty data at month ' + i)
		#to make gif using imageMagik, use terminal command: 
		#convert -loop 0 -delay 40 time_0.png time_1.png time_2.png time_3.png time_4.png time_5.png time_6.png time_7.png time_8.png time_9.png time_10.png time_11.png time_12.png time_13.png time_14.png time_15.png time_16.png time_17.png time_18.png time_19.png time_20.png time_21.png time_22.png time_23.png time.gif

def getData(num):
	import data_month

	X,Y,sevStr,Dtime = data_month.data_month(num)

	#lookup of numberal equivelant (should be done in a dict really)
	primary = ['ARSON', 'ASSAULT', 'BATTERY','BURGLARY','CONCEALED CARRY LICENSE VIOLATION','CRIM SEXUAL ASSAULT','CRIMINAL DAMAGE','CRIMINAL TRESPASS','DECEPTIVE PRACTICE','GAMBLING','HOMICIDE','HUMAN TRAFFICKING','INTERFERENCE WITH PUBLIC OFFICER','INTIMIDATION','KIDNAPPING','LIQUOR LAW VIOLATION','MOTOR VEHICLE THEFT','NARCOTICS','OBSCENITY','OFFENSE INVOLVING CHILDREN','OTHER NARCOTIC VIOLATION','OTHER OFFENSE','PROSTITUTION','PUBLIC INDECENCY','PUBLIC PEACE VIOLATION', 'ROBBERY','SEX OFFENSE','STALKING','THEFT','WEAPONS VIOLATION','NON-CRIMINAL','NON - CRIMINAL','NON-CRIMINAL (SUBJECT SPECIFIED)']
	severity = [0.5,0.72,0.89,0.44,0.11,0.94,0.28,0.11,0.44,0.11,0.94,1.0,0.11,0.22,1.0,0.11,0.17,0.22,0.11,0.89,0.22,0.11,0.17,0.11,0.22,0.94,0.94,0.67,0.17,0.22,0,0,0]

	sev = np.zeros(len(sevStr))
    
	#assign severity str to numerical value
	for i in range(len(sevStr)):
		ind = primary.index(sevStr[i])
		sev[i] = severity[ind]
	return X, Y, sev

def makeHeatmap(X, Y, sev, data_reduce):
	#reduce data size
	X = X/data_reduce
	Y = Y/data_reduce

	#find min and 'width' for image space
	x_min = np.min(X)
	y_min = np.min(Y)
	x_len = np.max(X) - x_min
	y_len = np.max(Y) - y_min

	x_min = 1094.231
	y_min = 1813.91
	x_len = 110.886
	y_len = 137.625

	#shift to [0,0] to save figure space
	x_shift = X - x_min
	y_shift = Y - y_min

	#define image size
	heatIm =  np.zeros([int(np.ceil((y_len+1))), int(np.ceil((x_len+1)))], dtype=np.uint16)

	#sum severities for image locations, n.b. x axis has to be flipped to match image coordinates
	for i in range(len(X)):
		heatIm[abs(int(y_shift[i])-heatIm.shape[0]+1), int(x_shift[i])] += int(sev[i]*10)

	#plot and save image
	plt.imshow(heatIm,cmap='hot')
	plt.xticks([])
	plt.yticks([])
	return heatIm, x_min, y_min

def getSev(x1, y1, heatIm, data_reduce, x_min, y_min):  #x and y are point from original scaling from chicago data set
	x1 = x1/data_reduce
	y1 = y1/data_reduce

	x1_shift = int(np.round(x1 - x_min))
	y1_shift = int(np.round(y1 - y_min))

	sev1 = heatIm[y1_shift,x1_shift]
	maxSev = np.max(heatIm)
	sevNorm = sev1/maxSev

	return sevNorm

def heatmapData(heatIm):    #converts pixels to crime data points of the heatmap
	data_points = sum(sum(heatIm))   #number of crime units
	heatData = np.zeros([data_points,2])
	count = 0
	for i in range(heatIm.shape[0]):
		for j in range(heatIm.shape[1]):
			units = heatIm[i,j]
			for k in range(units):
				heatData[count,0] = i
				heatData[count,1] = j
				count += 1
	return heatData

if __name__ == "__main__":
 	main()