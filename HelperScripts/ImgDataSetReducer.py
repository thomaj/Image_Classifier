import csv

# Reduces the full data set to retrieve only the rows in the CSV file
# containing all of the file names and their corresponding class number
# that are between the class numbers defined below.  For example, low of
# 83 and high of 97 retrieves all of the row in the full data csv (Image_
# Names_With_Classes.csv) that have a class in the Dodge category.  This
# creates a new csv file containing only these rows.

LOW_CLASS_NUM = 83
HIGH_CLASS_NUM = 97

with open('../Image_Names_With_Class.csv', 'rb') as f:
    reader = csv.reader(f)
    totalData = list(reader)

# Get only the image filename when the class is valid
reducedData = []
for elem in totalData:
    c = int(elem[1])
    if LOW_CLASS_NUM <= c and c <= HIGH_CLASS_NUM:
        reducedData.append(elem)

# Write the filename and class csv
with open('Reduced_Data.csv', 'wb') as toWrite:
    wr = csv.writer(toWrite, quoting=csv.QUOTE_ALL)
    wr.writerows(reducedData)


