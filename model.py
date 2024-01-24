from keras.models import Model
from keras import Input, layers
from keras.layers import concatenate
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, SeparableConv2D 
from tensorflow.keras.layers import BatchNormalization,Activation, Subtract, SpatialDropout2D, Flatten, UpSampling2D, MaxPooling2D


def SEDNet(input_size, num_classes):
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias = True)(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias = True)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias = True)(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias = True)(conv2)
 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',use_bias = True)(conv3)
    
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(conv4)

    up1 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(UpSampling2D(size = (2,2))(conv4))
    merge1 = concatenate([conv2,up1], axis = 3)  
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(merge1)
    conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(conv5)
    
    up2 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(UpSampling2D(size = (2,2))(conv5))
    merge2 = concatenate([conv1,up2], axis = 3)   
    conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(merge2)
    conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', use_bias = True)(conv6)
    
    conv7 = Conv2D(num_classes, (1,1), activation = 'sigmoid',use_bias = True)(conv6)
  
    return Model(inputs = input_size, outputs = conv7)

