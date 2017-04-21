import os
import csv
import sys
from PIL import Image

# Reduces the full image data set to contain only the files as defined in
# a CSV file in 'file_class_csv'.  This gets only those files and processes
# the image before saving it to a new directory.  The new image size is
# 224 x 224 pixels.
# If the augment argument is 'True', then the image
# set will be augmented.  This is done by resizing the original image to
# 256 x 256 and then taking 5 images of size 224 X 224 in each corner and
# in the center of the image.
# This also outputs a new CSV file containing each file name and its
# corresponding class.  This will be the same as before if no augmentation
# is done.


# if this input is 'True', then augments the data
augment_data_size = sys.argv[1]

# Define the file paths and directories
root_dir = os.path.abspath('..') # because in HelperScripts directory
images_dir = os.path.join(root_dir, 'cars_train')
output_dir = os.path.join(root_dir, 'HelperScripts/Resized_Images')
file_clases_csv = os.path.join(root_dir, 'Smaller_Data/Reduced_Data_Dodge.csv')
new_file_name_classes = os.path.join(root_dir, 'HelperScripts/newFileNamesAndClasses.csv')

FINAL_IMG_SIZE = (128, 128)

# For augmenting the data
IMAGE_SIZE = (256, 256)
boxes = [
    (0, 0, 224, 224),
    (32, 0, 256, 224),
    (0, 32, 224, 256),
    (32, 32, 256, 256),
    (16, 16, 240, 240)
]
letters = ['a', 'b', 'c', 'd', 'e']

# Create new list to hold the new images if the data is being augmented
filesClassesWithNewFiles = []

# Read in the files and corresponding classes
with open(file_clases_csv, 'rb') as f:
    reader = csv.reader(f)
    fileClasses = list(reader)


# Go through each image and resize so they are all the same size
if not os.path.exists(output_dir):
    os.makedirs(output_dir)     # Make the directory to store the images
    for fileClass in fileClasses:
        # Open the file with the name file name
        fName = fileClass[0]
        im = Image.open(images_dir + '/' + fileClass[0])
        r = im.resize(IMAGE_SIZE)
        if augment_data_size == 'True':
            # Want to get 5 images from this one image with size 224x224
            # Loop through boxes and crop and save each one
            for box, letter in zip(boxes, letters):
                newFileName = fName.split('.')[0] + letter + '.jpg'
                cropped = r.crop(box)   # Get the cropped out image
                croppedSmall = cropped.resize(FINAL_IMG_SIZE)
                croppedSmall.save(output_dir + '/' + newFileName, 'JPEG')
                newClass = int(fileClass[1]) - 83    # Dodge cars start at 83
                filesClassesWithNewFiles.append([newFileName, newClass])
        else:
            # Not augmenting, so resize orginal smaller and save
            smaller = r.resize((128, 128))
            smaller.save(output_dir + '/' + fName, 'JPEG')
            filesClassesWithNewFiles.append(fileClass)

        im.close()

    # Have gone through all images, create the new csv file
    with open(new_file_name_classes, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(filesClassesWithNewFiles)

else:
    print str(output_dir) + ' directory already exists'