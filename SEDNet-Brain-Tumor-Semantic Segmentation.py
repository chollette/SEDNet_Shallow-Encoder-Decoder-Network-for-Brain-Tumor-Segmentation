import os
import cv2
import glob
import h5py
import keras
import random                        
import numpy as np 
import pandas as pd
import nibabel as nib
from PIL import Image
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
from keras import Input, layers
from keras.utils import np_utils
from keras.layers import concatenate
from tensorflow.keras import layers, models
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import directed_hausdorff
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input,Concatenate, Conv2D,Conv2DTranspose, SeparableConv2D 
from tensorflow.keras.layers import BatchNormalization,Activation, Subtract, SpatialDropout2D, Flatten, UpSampling2D, MaxPooling2D

path = r'D:\My Documents\projects\mri\BraTS2020_TrainingData - withzero\MICCAI_BraTS2020_TrainingData - Copy - Copy'
os.chdir(path)

drs = glob.glob('*')


# # Data
from preprocessor import remove_files
from preprocessor import check_fileLen
from preprocessor import keep_fewSamples

# Remove all empty and non-tumor slices in the list
files_to_remove1 = remove_files(drs)
for file_path in files_to_remove:
    os.remove(file_path)

# Retrieve the length of the least length of folder 
perFoldLen = check_fileLen(drs)
print(np.min(perFoldLen))

# Retain only min slices of all folders
files_to_remove2 = keep_fewSamples(drs)
for file_path in files_to_remove2:
    os.remove(file_path)

# Retrieve the remaining slices per sample
mask_array = []
mask_name = []
img_data = []

file1 = 'mask'
file2 = 'flair'
for folder in drs:
    if 'BraTS20_Training' in folder:
        count = 0
        for i in os.listdir(path + '/' + folder + '/' + file1):
            im_path1 = os.path.join(folder+'/'+file1,i)
            im_path2 = os.path.join(folder+'/'+file2,i)
            image1 = cv2.imread(im_path1, cv2.COLOR_BGR2RGB)
            image2 = cv2.imread(im_path2, cv2.COLOR_BGR2GRAY)
            image2 = image2.astype('uint8')
            image2 = np.array(image2)
            img_data.append(image2)
            image1 = image1.astype('uint8')
            image1 = np.array(image1)
            mask_array.append(image1)
            mask_name.append(i)
            count+=1
label_data = np.array(mask_array)
train_data = np.array(img_data)                


"""def augmentation(X, Y, seed, batch_size):
    
    data_gen = dict(rotation_range =15,   
                             zoom_range=0.2, 
                             horizontal_flip = True, 
                             vertical_flip = True) 

    image_datagen = ImageDataGenerator(**data_gen)
    mask_datagen = ImageDataGenerator(**data_gen)
    
    Y = np.array(Y)
    Y = Y.reshape((1,) + Y.shape) #for 3 channel image
    X = np.array(X)
    X = X.reshape((1,) + X.shape + (1,))

    image_datagen.fit(X, augment=True, seed=seed)
    mask_datagen.fit(Y, augment=True, seed=seed)

    image_gen = image_datagen.flow(X,Y, batch_size=batch_size, seed=seed)
    mask_gen = mask_datagen.flow(Y,X, batch_size=batch_size, seed=seed)
    k=0
    while True:
        re_x = image_gen.next()
        re_y = mask_gen.next()
        k+=1
        if k == 5:
            break
        
    return re_x[0], re_y[0]"""


# ### Create a HDF5 file
#Single channel 
# creat h5py data - supports large datasets, larger than RAM size
# create HDF5 file
#with h5py.File('braintumordataset_singch.hdf5', 'w') as hdb:
#    images = hdb.create_dataset('X_images', data=train_data, shape=(8487, 128, 128), compression='gzip', chunks=True)
#    mask = hdb.create_dataset('Y_mask', data=label_data, shape=(8487, 128, 128, 3), compression='gzip', chunks=True)

# read HDF5 file
hbd = h5py.File('data/BraTS2020_singchan.hdf5', 'r')


XX_images = hbd.get('X_images')[:]
YY_mask = hbd.get('Y_mask')[:]

print(XX_images.shape)
print(YY_mask.shape)

#close file
hbd.close()

np.max(XX_images[0])


# ###  Data Split
#Split dataset
# Split the data into training and testing sets\n",
X_train, X_VT, y_train, y_VT = train_test_split(XX_images, YY_mask, test_size = 0.2, random_state = 0)
X_val, X_test, y_val, y_test = train_test_split(X_VT, y_VT, test_size=0.5, random_state=0)

X_train = np.array(X_train).astype('float32')
X_train /=255.0
X_val = np.array(X_val).astype('float32')
X_val/=255.0
X_test = np.array(X_test).astype('float32')
X_test/=255.0

y_train = np.array(y_train).astype('float32')
y_train/=255.0
y_val = np.array(y_val).astype('float32')
y_val/=255.0
y_test = np.array(y_test).astype('float32')
y_test/=255.0

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)


# ### Data Generator
keras = tf.compat.v1.keras
Sequence = keras.utils.Sequence

class MRISeq(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        #elf.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return batch_x, batch_y 

train_gen = MRISeq(X_train, y_train, batch_size=23)
val_gen = MRISeq(X_val, y_val, batch_size=23)
test_gen = MRISeq(X_test, y_test, batch_size=23)


# ### Visualize
lent = []
count=0
for a, b in train_gen:
    lent.append(a)   
    images = a
    count+=1
    for i in range(len(images)):
        plt.imshow(images[i])
        plt.colorbar()
        plt.show()
    break
print(images.shape)

print(np.array(lent).shape)


# # Model
from model import SEDNet
IMG_SIZE=128
input_size = Input((IMG_SIZE, IMG_SIZE,1))
num_classes = 3
model = SEDNet(input_size, num_classes)
model.summary()


# # Transfer Learning
from evaluation_metrics import dice_NTC
from evaluation_metrics import dice_ED
from evaluation_metrics import dice_ET
from loss_function import bce_softDice_loss2

model = r"..\SEDNet-singch_0050.h5"
keras_callbacks   = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=2, min_lr=0.0000000000000001, verbose=1)]

SEDNet = keras.models.load_model(model, compile=False)
SEDNet.compile(loss=bce_softDice_loss2, optimizer=keras.optimizers.Adam(learning_rate=0.0003), metrics = [dice_coef, dice_NTC, dice_ED, dice_ET])
 
#freeze first layers, until 19th layer
for i in range(19,21):
    SEDNet.layers[i].trainable = True

x = SEDNet.layers[21].output
SEDNetX = Model(inputs=SEDNet.input, outputs=x)

score = SEDNet.evaluate(test_gen, batch_size=100, callbacks= keras_callbacks)
print(score)
score = SEDNetX.evaluate(test_gen, batch_size=100, callbacks= keras_callbacks)
print(score)


# # Evaluation

# ### Hausdorff distance
from evaluation_metrics import hausdorff_NTC
from evaluation_metrics import hausdorff_ED
from evaluation_metrics import hausdorff_ET

hdNTC = []
hdED = []
hdET = []
IMG_SIZE = 128
X = np.empty((1,IMG_SIZE, IMG_SIZE, 1))
y = np.empty((1, IMG_SIZE, IMG_SIZE, 3))

for i in range(len(X_test)):
    X[0,:,:,0] = X_val[i]
    y[0,:,:,:] = y_val[i]
    p = modell.predict(X, verbose=0)
    a, b, c = Hausdorff_distance(y[0,:,:,:], p[0,:,:])
    hdNTC.append(a)
    hdED.append(b)
    hdET.append(c)
print(np.mean(hdNTC))
print(np.mean(hdED))
print(np.mean(hdET))


# ### Visual Evaluation on Brats2020 testing set
IMG_SIZE = 128

X = np.empty((1,IMG_SIZE, IMG_SIZE, 1))
y = np.empty((1, IMG_SIZE, IMG_SIZE, 3))

X[0,:,:,0] = X_test[170]
y[0,:,:,:] = y_test[170]
y_pred = SEDNetX.predict(X) 

plt.imshow(y_pred[0,:,:])
plt.imshow(y[0,:,:,:])


# ### Visual Evaluation on Brats2020 Validation
IMG_SIZE = 128
case_path = r"D:\My Documents\projects\mri\BraTS2020_TrainingData - withzero\BraTS2020_ValidationData\MICCAI_BraTS2020_ValidationData\BraTS20_Validation_040\flair\BraTS20_Validation_040_70.jpg"

X = np.empty((1,IMG_SIZE, IMG_SIZE, 1))

X[0,:,:,0] = cv2.imread(case_path,  cv2.COLOR_BGR2GRAY)
X = np.array(X).astype('float32')
X/=255.0

y_pred = SEDNetX.predict(X) 
plt.imshow(y_pred[0,:,:])

