import scipy.io as sp

# Extracts the file names and the corresponding class number from
# the .mat file definied below.  Then creates a CSV file with each
# row having first the file name and then the class number

# Load in the .mat data
car_annotations = sp.loadmat('../devkit/cars_train_annos.mat')

# All we should need to care about is the class number and the fname
# ['annotations'][row][4][0][0] gives the class number
# ['annotations'][row][5][0] gives the image file name



fileToWrite = open('Image_Names_With_Class.csv', 'w')
for row in car_annotations['annotations'][0]:
    classNum = row[4][0][0]
    fileName = row[5][0]

    fileToWrite.write(fileName + ', ' + str(classNum) + '\n')

print 'Done!'