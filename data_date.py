import pandas
import numpy as np
from tqdm import tqdm

#define column names
colnames = ['ID', 'Case Number', 'Date', 'Block', 'IUCR','PrimaryType','Description','Location Description','Arrest','Domestic','Beat','District','Ward','Community','FBI Code','XCoordinate','YCoordinate','Year','Updated On','Latitude','Longitude','Location']
data = pandas.read_csv('crimes2016.csv', names=colnames)  #extract data

#extract useful columns to lists
X = data.Date.tolist()