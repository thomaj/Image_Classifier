import scipy.io as sp

# Extracts the classes from the .mat file as defined below
# Then creates a CSV file containing the classes

# Load in the .mat data
classes = sp.loadmat('../devkit/cars_meta.mat')

fileToWrite = open('../Classes.csv', 'w')

# ['class_names'][0][row][0] gives the class name (car name)
for row in classes['class_names'][0]:
    fileToWrite.write(row[0] + '\n')


print 'Done!'