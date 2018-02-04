import numpy as np 
import matplotlib.pyplot as plt 
from collections import Counter

Beats = np.load('beats.npy')    #load beats data for missing data
BeatsClean = np.load('beatsClean.npy')  #Load beats data for NON missing data
sevStr = np.load('severity2.npy')        #load severity for missing data
sevStrClean = np.load('severity.npy')    #load severity for NON missing data


#lookup of numberal equivelant (should be done in a dict really)
primary = ['ARSON', 'ASSAULT', 'BATTERY','BURGLARY','CONCEALED CARRY LICENSE VIOLATION','CRIM SEXUAL ASSAULT','CRIMINAL DAMAGE','CRIMINAL TRESPASS','DECEPTIVE PRACTICE','GAMBLING','HOMICIDE','HUMAN TRAFFICKING','INTERFERENCE WITH PUBLIC OFFICER','INTIMIDATION','KIDNAPPING','LIQUOR LAW VIOLATION','MOTOR VEHICLE THEFT','NARCOTICS','OBSCENITY','OFFENSE INVOLVING CHILDREN','OTHER NARCOTIC VIOLATION','OTHER OFFENSE','PROSTITUTION','PUBLIC INDECENCY','PUBLIC PEACE VIOLATION', 'ROBBERY','SEX OFFENSE','STALKING','THEFT','WEAPONS VIOLATION','NON-CRIMINAL']
severity = [0.5,0.72,0.89,0.44,0.11,0.94,0.28,0.11,0.44,0.11,0.94,1.0,0.11,0.22,1.0,0.11,0.17,0.22,0.11,0.89,0.22,0.11,0.17,0.11,0.22,0.94,0.94,0.67,0.17,0.22,0]

sev = np.zeros(len(sevStr))
sevClean = np.zeros(len(sevStrClean))

#assign severity str to numerical value
for i in range(len(sevStr)):
	ind = primary.index(sevStr[i])
	sev[i] = severity[ind]
    
for i in range(len(sevStrClean)):
	ind = primary.index(sevStrClean[i])
	sevClean[i] = severity[ind]

tally = np.zeros(50)

for i in range(len(Beats)):
    for j in range(50):
        if Beats[i]==j:
            tally[j]+=1

count = Counter(Beats)       
keys = count.keys()
print(count.values())

countClean = Counter(BeatsClean)       
keysClean = countClean.keys()


plt.bar(list(keys), count.values())
plt.title('Wards with data missing')
plt.xlabel('Ward')
plt.ylabel('Number of crimes')
plt.show()

plt.bar(list(keysClean), countClean.values())
plt.title('Wards without data missing')
plt.xlabel('Ward')
plt.ylabel('Number of crimes')
plt.show()

