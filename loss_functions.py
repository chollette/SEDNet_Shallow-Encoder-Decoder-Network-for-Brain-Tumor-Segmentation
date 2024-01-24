import keras.backend as K
from keras.losses import binary_crossentropy, BinaryCrossentropy

# loss for a binary class

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

#
def soft_dice(y_true, y_pred, axis=(0, 1, 2), epsilon=0.00001):
    dice_numerator = (2 * K.sum(y_pred * y_true, axis = axis)) + epsilon
    dice_denominator = K.sum(y_pred * y_pred, axis = axis) + K.sum(y_true * y_true, axis=axis) + epsilon
    softDice_loss = 1 - K.mean(dice_numerator/dice_denominator)
    return softDice_loss
	
def bce_softDice_loss1(y_true, y_pred): # equal weighting
    w1 = 0.5
    w2 = 0.5
    l = w1 * binary_crossentropy(y_true, y_pred) + w2 * softDice_loss(y_true,y_pred)
    return l  

def bce_softDice_loss2(y_true, y_pred): #priority weighting
    w1 = 0.6
    w2 = 0.4
    l = w1 * binary_crossentropy(y_true, y_pred) + w2 * softDice_loss(y_true,y_pred)
    return l  

def bce_sdice_loss(y_true, y_pred): #no weighting
    l = binary_crossentropy(y_true, y_pred) + softDice_loss(y_true,y_pred)
    return l 

