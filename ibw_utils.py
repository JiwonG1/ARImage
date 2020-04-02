from pycroscopy.io.translators.igor_ibw import IgorIBWTranslator as igor
import numpy as np
import h5py
import imutils
import cv2
import pyUSID as usid
from skimage import filters
from scipy.optimize import curve_fit
from tqdm import tqdm
from pyprnt import prnt
import matplotlib.pyplot as plt

""" Channel name 
Channel_000 : HeightTrace (m) 
Channel_001 : HeightRetrace (m)
Channel_002 : AmplitudeTrace (m) 
**Channel_003 : AmplitudeRetrace (m)
Channel_004 : PhaseTrace (m)
**Channel_005 : PhaseRetrace (deg)
Channel_006 : LateralTrace (deg) 
Channel_007 : LateralRetrace (m)
Channel_008 : UserCalcTrace (m) 
Channel_009 : UserCalcRetrace (m)
Channel_010 : UserCalcBTrace (m)
Channel_012 : LatAmplitudeTrace (m)
**Channel_013 : LatAmplitudeRetrace (m)
Channel_014 : LatPhaseTrace (m)
**Channel_015 : LatPhaseRetrace (m) 
Channel_016 : HeightRetrace (m) ==> Flatten height
"""

class AngleImage(object):
    
    def __init__(self, imagename, data):
        self.imagename = imagename
        self.namedangle = int(imagename.split('_')[-1])
        self.angle = 0
        self.data = data
        self.trans = (0, 0)
        self.alignimg = None
        self.piezo = None
        self.Normpiezo = None

    def __repr__(self):
        return repr((self.angle, self.imagename))

    def SaveBinaryImg(self, img, cut=False, invert=False, response=False):
        height, width = img.shape if len(img.shape) == 2 else img.shape[:2]
        h, w = int(height * (0.5/np.sqrt(2))), int(width * (0.5/np.sqrt(2)))
        img1 = img[int(height/2-h):int(height/2+h),int(width/2-w):int(width/2+w)]
        val = filters.threshold_otsu(img1)
        if cut:
            img = img1
        if invert:
            img = img > val
        else:
            img = img < val
        img = img.astype(np.uint8)
        if not response:
            img *= 255
        self.binaryimg = img
        return img

    def ChannelList(self):
        return channel_list(self.data)

    def ChannelName(self, channel):
        return h5toChannelName(self.data, channel)

    def Piezoresponse(self, phasechannel, ampchannel, fix=False):
        phaseimg = h5toimg(self.data, phasechannel)[0]
        ampimg = h5toimg(self.data, ampchannel)[0]
        piezoresponse = ampimg * np.cos(phaseimg*np.pi/180)
        if fix:
            piezoresponse = imutils.translate(piezoresponse, self.trans[0], self.trans[1])
            piezoresponse = imutils.rotate(piezoresponse, self.angle)
        #self.piezo = piezoresponse.astype(np.uint8)
        self.piezo = piezoresponse
        return self.piezo
    
    def NormPiezoresponse(self, phasechannel, ampchannel, fix=False):
        phaseimg = h5toimg(self.data, phasechannel)[0]
        ampimg = h5toimg(self.data, ampchannel)[0]
        Normpiezo = ampimg * np.cos(phaseimg*np.pi/180) / np.max(ampimg)
        #Normpiezo = np.zeros(piezo.shape)
        #Normpiezo = cv2.normalize(piezo, Normpiezo, 0, 255, cv2.NORM_MINMAX)
        if fix:
            Normpiezo = imutils.translate(Normpiezo, self.trans[0], self.trans[1])
            Normpiezo = imutils.rotate(Normpiezo, self.angle-90)
        self.Normpiezo = Normpiezo
        #print(np.max(ampimg/np.max(ampimg)))
        #print(np.min(ampimg/np.max(ampimg)))
        return self.Normpiezo

def channel_list(h5):
    channel = {}
    for group in h5:
        if group[:5] == 'Chann':
            channel[group] = usid.hdf_utils.get_attr(h5[group+'/Raw_Data'], 'quantity')
    return channel

def channel_list_folder(h5Files):
    channels = channel_list(h5Files[0].data)
    #h5Filenames = [h5.imagename for h5 in h5Files]
    if len(h5Files) == 1:
        return 1, channels
    else:
        channel_type = {h5Files[0].imagename:channels}
        type_files = [h5Files[0].imagename]
        type_dict =  {len(type_files):h5Files[0].imagename}
        for i in range(1,len(h5Files)):
            if not channel_list(h5Files[i].data) in channel_type.values():
                channel_type[h5Files[i].imagename] = channel_list(h5Files[i].data)
                type_files.append(h5Files[i].imagename)
                type_dict[len(type_files)] = h5Files[i].imagename
            else:
                for key, val in channel_type.items():
                    if val == channel_list(h5Files[i].data):
                        ChannelType = type_files.index(key) + 1
                type_dict[ChannelType] = type_dict[ChannelType] + ', ' + h5Files[i].imagename
        if len(type_files) > 1:
            return len(type_files), (type_dict, type_files, channel_type)
        else:
            return 1, channel_type[h5Files[0].imagename]

def ibw2h5(ibw, exist=False):
    if exist:
        h5 = h5py.File(ibw, mode='r')
    else:
        translator = igor()
        h5_path = translator.translate(ibw, verbose=False)
        h5 = h5py.File(h5_path, mode='r')
    return h5['Measurement_000/']

def h5toimg(h5, channel):
    cmap = usid.hdf_utils.get_attr(h5, 'ColorMap '+str(channel))
    channel = 'Channel_' + str(channel).zfill(3)
    img = usid.USIDataset(h5[channel+'/Raw_Data'])
    height, width = getDim(h5, channel)[:2]
    img = np.flip(np.reshape(img, (height, width)), axis=0)
    img = ((img+abs(np.amin(img)))/(np.amax(img)-np.amin(img))*255)
    return img.astype(np.uint8), ColorMap(cmap)

def ibw2img(ibw, channel):
    return h5toimg(ibw2h5(ibw), channel)

def h5toChannelName(h5, channel):
    channels = channel_list(h5)
    return channels['Channel_'+str(channel).zfill(3)]

def ImageFlatten(img, order):

    if len(img.shape) != 2:
        print('Image dimension is not 2D.')
        return None
    
    scanlines, scanpoints = img.shape
    
    if order == 0:
        flattenimg = img[0,:] - np.mean(img[0,:])
        for i in range(1, scanlines):
            flattenimg = np.vstack((flattenimg, img[i,:]-np.mean(img[i,:])))
    else:
        x = np.arange(1, scanpoints+1)
        fit = np.polyfit(x, img[0,:], order)
        poly = np.poly1d(fit)
        flattenimg = img[0,:] - poly(x)
        for i in range(1, scanlines):
            fit = np.polyfit(x, img[i,:], order)
            poly = np.poly1d(fit)
            flattenimg = np.vstack((flattenimg, img[i,:] - poly(x)))

    return flattenimg
### Under Measurement_000/
## BaseName
## ColorMap (Channel) #
## Display Offset #
## Display Range #
## Scan size

def ColorMap(cmap):
    cmap_dict = {'Grays256':'gray', 'VioletOrangeYellow':'PuOr'}
    return cmap_dict[cmap]

def getDim(h5, channel):
    h5 = h5[channel+'/Raw_Data']
    inds = usid.hdf_utils.get_auxiliary_datasets(h5, ['Position_Indices'])[0]
    pos_dim_sizes = usid.hdf_utils.get_dimensionality(inds)
    pos_dim_names = usid.hdf_utils.get_attr(inds, 'labels')
    dim = {}
    for name, length in zip(pos_dim_names, pos_dim_sizes):
        dim[name] = length
    X_dim, Y_dim = dim['X'], dim['Y']
    units = usid.hdf_utils.get_attr(h5, 'units')
    return X_dim, Y_dim, units

def getAttr(h5, attr):
    return usid.hdf_utils.get_attr(h5, attr)

def getAttrList(h5_main):

    ## Return Attributes dictionary

    # Initial ScanSize
    # Initial ScanAngle
    # Initial ScanLines
    # Initial ScanPoints
    # Initial ScanRate
    # ImageNote

    Attrs = {}
    for key, val in usid.hdf_utils.get_attributes(h5_main).items():
        Attrs[key] = val
    return Attrs

def _Cosfunc(x, v, theta):
    return v * np.cos((x-theta)*np.pi/180)

def AngleMap(h5Files):
    
    x_data = [h5.angle for h5 in h5Files]
    y_dataStack = [h5.Piezoresponse(15, 13, fix=True)/255 for h5 in h5Files]
    y_dataStack = np.stack(y_dataStack, axis=-1)
    sx, sy, _ = y_dataStack.shape 
    #img = h5Files[6].Piezoresponse(15, 13, fix=True)
    #y_dataStack = np.stack(y_dataStack, axis=-1)
    #print(h5Files[0].piezo)
    #for idx in range(1, len(h5Files)):
    #    x_data.append(h5Files[idx].angle)
    #    img = h5Files[idx].piezo.shape
    #    y_dataStack = np.dstack((y_dataStack, img))
    #    y_data.append(piezoresponse[234, 236]/255)
    #ParamV, ParamTheta = np.empty((sx, sy)), np.empty((sx, sy))
    #print(x_data)
    #print(y_dataStack[35, 235, :])
    #print(curve_fit(_Cosfunc, x_data, y_dataStack[35, 235, :])[0])
    #ParamV[35, 235], ParamTheta[25, 235] = curve_fit(_Cosfunc, x_data, y_dataStack[x, y, :])[0]
    ## Not kernel operating
    #for i in tqdm(range(sx*sy)):
    #    x, y = divmod(i, sx)
    #    ParamV[x, y], ParamTheta[x, y] = curve_fit(_Cosfunc, x_data, y_dataStack[x, y, :])[0]

    ## Apply Kernel 3*3 pixel, average pooling
    Xkernel_size = 3
    Xwindow = int((Xkernel_size-1)/2)
    Ykernel_size = 3
    Ywindow = int((Ykernel_size-1)/2)
    kernel_sx = divmod(sx, Xkernel_size)[0]
    kernel_sy = divmod(sy, Ykernel_size)[0]
    VectorOrigins = [(x, y) for x in range(1, kernel_sx+1) for y in range(1, kernel_sy+1)]
    VectorDict = {}
    for x, y in tqdm(VectorOrigins):
        coord = (x*Xkernel_size, y*Ykernel_size)
        #print(coord[0]-Xwindow)
        #print(coord[0]+Xwindow)
        average_data = y_dataStack[coord[0]-Xwindow-1:coord[0]+Xwindow,coord[1]-Ywindow-1:coord[1]+Ywindow,:]
        #print(average_data)
        #print(average_data.shape)
        average_data = average_data.sum(axis=0)
        #print(average_data.shape)
        average_data = average_data.sum(axis=0)
        #print(average_data.shape)
        #print(average_data)
        VectorDict[coord] = curve_fit(_Cosfunc, x_data, average_data)[0]
    
    prnt(VectorDict)
    asdf
    print(ParamTheta)
    print(ParamTheta.shape)
    cv2.imwrite('./theat.jpg', ParamTheta)
    plt.imshow(ParamTheta, origin='lower', interpolation='none', cmap='gray')    
    clb = plt.colorbar(shrink=0.9)
    clb.set_label('Degree')
    plt.savefig('./theta.jpg')
    asdf
    param = np.apply_along_axis(func1d, axis=2, arr=y_dataStack)
    print(param)
    asdf
    print(y_dataStack[5])
    print(y_dataStack[2].shape)
    return y_dataStack[3]
    #print(x_data)
    #print(y_data)
    #popt, _ = curve_fit(_Cosfunc, x_data, y_data)
    #print(popt)
    #for x, y in zip(range(size[0]), range(size[1])):
        