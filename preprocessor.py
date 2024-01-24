import os
import cv2
import glob
import numpy
import keras
import numpy as np
import nibabel as nib
from PIL import Image
from skimage import exposure 
import matplotlib.pyplot as plt

#Rectangular Kernel
Rkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

path = r'your directory'
os.chdir(path)

drs = glob.glob('*') 

IMG_SIZE = 128
VOLUME_SLICES = 155
VOLUME_START_AT = 0

# Rectangular Kernel
Rkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

file1 = 'flair'
file2 = 'mask'

def extract_flair(list_of_folders):
    for file in drs:
        if 'BraTS20_Training' in file:
            pa = os.path.join(file, 'flair')
            if not os.path.exists(pa):
                os.mkdir(pa)
                # if it does exist, retrieve the path for each slice
                for i in os.listdir(path + '/' + file):
                    case_path = os.path.join(path, file)
                    data_path = os.path.join(case_path , f'{file}_flair.nii')       
                    flair = nib.load(data_path).get_fdata()         
                    count = 0
                    for j in range(VOLUME_SLICES):#
                        img = np.array(cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)))
                        img = cv2.normalize(img, None, norm_type=cv2.NORM_MINMAX)
                        img = (img*255).astype(np.uint8)
                        filename = path + '/' + file +'/flair/' + file + '_' + str(count) + '.jpg'
                        cv2.imwrite(filename, img)					  
                        count+=1
	
    
def extract_mask(list_of_folders):
	for file in drs:
		if 'BraTS20_Training' in file:
			pa = os.path.join(file, 'mask')
			if not os.path.exists(pa):
				os.mkdir(pa) 
			# if it does exist, retrieve the path for each slide
			for i in os.listdir(path + '/' + file):
				case_path = os.path.join(path, file)
				data_path = os.path.join(case_path , f'{file}_seg.nii') 
				seg = nib.load(data_path).get_fdata()
				count = 0
				for j in range(VOLUME_SLICES):                         
					y =  cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)) 
					all_classes = keras.utils.to_categorical(y, num_classes = 5)
					#WT = cv2.bitwise_not(all_classes[:,:,0])
					NTC = all_classes[:,:,1] 
					ED = all_classes[:,:,2]
					ET = all_classes[:,:,4]
					final_classes = np.dstack((NTC,ED,ET))*255 #NTC, ED, ET      
					filename = path + '/' + file +'/mask/' + file + '_' + str(count) + '.jpg'
					cv2.imwrite(filename, final_classes)
					count+=1


def remove_files(list_of_folders):
	files_to_remove1 = []  # List to store files that need to be removed
	for folder in list_of_folders:
		if folder.startswith('BraTS20_Training'):  # Check if folder name starts with 'BraTS20_Training'
			for i in os.listdir(path + '/' + folder + '/' + file2):
				im_path2 = os.path.join(folder, file2, i)
				im_path1 = os.path.join(folder, file1, i)
				im = cv2.imread(im_path2)
				im = im.astype(np.uint8)
				im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				openn = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
				close = cv2.morphologyEx(openn, cv2.MORPH_CLOSE, Rkernel, iterations=3)
				maxx = np.max(close)  # Find maximum value in 'close' array
				if maxx == 0:
					files_to_remove1.extend([im_path2, im_path1])  # Add files to the list
				else:
					_, contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
					cnt = max(contours, key=cv2.contourArea)  # Find contour with largest area directly
					area = cv2.contourArea(cnt)
					if area <= 50: 
						files_to_remove1.extend([im_path2, im_path1])  # Add files to the list
	return files_to_remove1
	
def check_fileLen(list_of_folders):
    # Initialize an empty list to store the counts
    total = []
    # Iterate over the directories
    for folder in list_of_folders:
        # Check if the folder name contains 'BraTS20_Training'
        if 'BraTS20_Training' in folder:
            # Initialize count to 0 for each folder
            count = 0
            # Get the path to the folder
            folder_path = os.path.join(path, folder, file2)
            count = len(os.listdir(os.path.join(folder_path)))
            # Append the count to the total list
            total.append(count)
    # Find the minimum count
    return total

def keep_fewSamples(list_of_folders):
	files_to_remove2 = []  # List to store files that need to be removed

	# Iterate over the directories
	for folder in drs:
		if 'BraTS20_Training' in folder:
			# Get the file paths that match the pattern
			file_paths = glob.glob(os.path.join(folder, file2, '*'))
			
			# Check if the number of files is greater than 23
			if len(file_paths) > 23:
				# Retain only 23 slices				
				im_path2 = os.path.join(folder, file2, os.path.basename(file_paths))
				im_path1 = os.path.join(folder, file1, os.path.basename(file_paths))
				files_to_remove2.extend([im_path2, im_path1])  # Add files to the list
	return files_to_remove2


	


