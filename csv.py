import csv

with open('crimes2016.csv', newline='') as csvfile:
	rea = csv.reader(csvfile)

print(type(rea))
print(reader[0,0])