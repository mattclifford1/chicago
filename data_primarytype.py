import pandas
import numpy as np
from tqdm import tqdm

#define column names
colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data

#extract useful columns to lists
X = data.PrimaryType.tolist()

#sort alphabetically the unique values of primary type list
Y = sorted(list(set(X)))

#get rid of first column containing string of column name
Y.remove('PrimaryType')
X.remove('PrimaryType')

#finding the index of a specific crime primary type
X = np.array(X)
find_item = np.where(X==Y[21])
index = find_item[0]

Z = data.Description.tolist()
Z.remove('Description')
for i in range(len(index)):
    Z[i] = Z[index[i]]

Z[i+1:]=[]
Z = sorted(list(set(Z)))