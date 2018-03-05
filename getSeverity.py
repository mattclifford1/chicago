import numpy as np 
import sys

P = np.load('severities.npy')

if len(sys.argv) > 3:
	print('too many input arguments, input X and Y only')
elif len(sys.argv) < 3:
	print('input both X and Y coordinates')
else:
	X = int(sys.argv[1])
	Y = int(sys.argv[2])

	xMax = int(1205117)
	yMax = int(1951535)

	if X > xMax or Y > yMax:
		print('coordinates off the grid')
	else:
		#need to reduce to 100 by 100 - the resolution that P was made at
		xRes = xMax/100
		yRes = yMax/100

		x = int(X/xRes)
		y = int(Y/yRes)

		print('severity at '+ str(X)+', '+str(Y)+' is: '+ str(P[x,y]))