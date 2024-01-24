import keras.backend as K

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)
	
	
def dice_NTC(y_true, y_pred, smooth=1.0):
    y_true = K.flatten(y_true[:,:,:,0])
    y_pred = K.flatten(y_pred[:,:,:,0])
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)


def dice_ED(y_true, y_pred, smooth=1.0):
    y_true = K.flatten(y_true[:,:,:,1])
    y_pred = K.flatten(y_pred[:,:,:,1])
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)


def dice_ET(y_true, y_pred, smooth=1.0):
    y_true = K.flatten(y_true[:,:,:,2])
    y_pred = K.flatten(y_pred[:,:,:,2])
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + smooth)
    

def Hausdorff_distance(y_true, y_pred):
    dim =len(y_true.shape)
    hd=[]
    for k in range(3):
        sym_hausdorff = max(directed_hausdorff(y_true[...,k],y_pred[...,k])[0], directed_hausdorff(y_pred[...,k],y_true[...,k])[0])
        hd.append(sym_hausdorff)
    hd = np.asarray(hd)
    return np.mean(hd[1:])


def hausdorff_NTC(y_true, y_pred):
    y_true = K.flatten(y_true[:,:,0])
    y_pred = K.flatten(y_pred[:,:,0])
    return Hausdorff_distance(y_true, y_pred)

def hausdorff_ED(y_true, y_pred):
    y_true = K.flatten(y_true[:,:,1])
    y_pred = K.flatten(y_pred[:,:,1])
    return Hausdorff_distance(y_true, y_pred)

def hausdorff_ET(y_true, y_pred):
    y_true = K.flatten(y_true[:,:,2])
    y_pred = K.flatten(y_pred[:,:,2])
    return Hausdorff_distance(y_true, y_pred)


