import numpy as np
import cv2

def ColorCode(Color):
    R, G, B = 255, 255, 255 
    ColorCode = {
            'red' : (R, 0, 0),
            'red_orange' : (R, 0.25*G, 0), 
            'orange' : (R, 0.5*G, 0), 
            'orange_yellow' : (R, 0.75*G, 0), 
            'yellow' : (R, G, 0), 
            'yello_yellowishgreen' : (0.75*R, G, 0),
            'yellowishgreen' : (0.50*R, G, 0), 
            'yellowishgreen_green' : (0.25*R, G, 0), 
            'green' : (0, G, 0), 
            'green_bluishgreen' : (0, G, 0.25*B),
            'bluishgreen' : (0, G, 0.50*B), 
            'bluishgreen_cyan' : (0, G, 0.75*B), 
            'cyan' : (0, G, B), 
            'cyan_processcyan' : (0, 0.75*G, B), 
            'processcyan' : (0, 0.50*G, B),
            'processcyan_blue' : (0, 0.25*G, B), 
            'blue' : (0, 0, B), 
            'blue_violet' : (0.25*R, 0, B), 
            'violet' : (0.50*R, 0, B), 
            'violet_magenta' : (0.75*R, 0, B), 
            'magenta' : (R, 0, B),
            'magenta_processmagenta' : (R, 0, 0.75*B), 
            'processmagenta' : (R, 0, 0.50*B), 
            'processmagenta_red' : (R, 0, 0.25*B)
    }
    return ColorCode[str(Color)]

def ColorOrder(ColorNumber):
    ColorOrderList = [
            'yellow', 'yello_yellowishgreen',
            'yellowishgreen', 'yellowishgreen_green', 'green', 'green_bluishgreen',
            'bluishgreen', 'bluishgreen_cyan', 'cyan', 'cyan_processcyan', 'processcyan',
            'processcyan_blue', 'blue', 'blue_violet', 'violet', 'violet_magenta', 'magenta',
            'magenta_processmagenta', 'processmagenta', 'processmagenta_red',
            'orange_yellow', 'orange', 'red_orange', 'red' 
            ]
    return ColorOrderList[int(ColorNumber-1)]

def ColorNumber(Color):
    ColorOrderList = [
            'red', 'red_orange', 'orange', 'orange_yellow', 'yellow', 'yello_yellowishgreen',
            'yellowishgreen', 'yellowishgreen_green', 'green', 'green_bluishgreen',
            'bluishgreen', 'bluishgreen_cyan', 'cyan', 'cyan_processcyan', 'processcyan',
            'processcyan_blue', 'blue', 'blue_violet', 'violet', 'violet_magenta', 'magenta',
            'magenta_processmagenta', 'processmagenta', 'processmagenta_red'
            ]
    return ColorOrderList.index(Color)+1

def ComplementaryColor(ColorNumber):
    return ColorOrder(ColorNumber + 12) if ColorNumber < 12 else ColorOrder(ColorNumber - 12)

def Coloring(img, colorcode):
    shape = (3, img.shape[0], img.shape[1])
    ColorImg = np.zeros(shape)

    # Fix to opencv bgr order
    ColorImg[0][img == True] = int(colorcode[2])
    ColorImg[1][img == True] = int(colorcode[0])
    ColorImg[2][img == True] = int(colorcode[1])
    ColorImg = np.rollaxis(ColorImg, 0, 3)
    
    b_channel, g_channel, r_channel = cv2.split(ColorImg)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255

    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

def ColorMapping(objlist, StartColorNumber):
    TurnWhite, TurnBlack = [], []
    for obj1, obj2 in zip(objlist[:-1], objlist[1:]):
        turnwhite, turnblack = Compare(obj1.binaryimg, obj2.binaryimg)
        WhiteColor = ColorOrder(StartColorNumber)
        BlackColor = ColorOrder(StartColorNumber+12)
        turnwhite = Coloring(turnwhite, ColorCode(WhiteColor))
        turnblack = Coloring(turnblack, ColorCode(BlackColor))
        TurnWhite.append(turnwhite)
        TurnBlack.append(turnblack)
        if StartColorNumber > 0:
            StartColorNumber += 1
        else:
            StartColorNumber -= 1
    return TurnWhite, TurnBlack

def Compare(img, img_ref):

    if img.shape != img_ref.shape:
        return None

    ref_white, ref_black = np.tile(False, img.shape), np.tile(False, img.shape)
    img_white, img_black = np.tile(False, img.shape), np.tile(False, img.shape)
    
    ref_white[img_ref == 255] = True
    ref_black[img_ref == 0] = True
    img_white[img == 255] = True
    img_black[img == 0] = True
    
    turn_white = np.logical_and(ref_black, img_white)
    turn_black = np.logical_and(ref_white, img_black)
    
    return turn_white, turn_black

def Blending(imgs):
    for idx in range(1, len(imgs)):
        img = imgs[idx-1]
        if idx == 1:
            first_img = img
            #continue
        else:
            second_img = img
            second_weight = 1/(idx+1)
            first_weight = 1 - second_weight
            first_img = cv2.addWeighted(first_img, first_weight, second_img, second_weight, 0)
    return first_img

def AdjustGamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    image = image.astype("uint8")
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def MergeImg(imglist):
    #ImageWeight = 1/len(imglist)
    MergeImg = imglist[0] #* ImageWeight
    for img in imglist[1:]:
        MergeImg = MergeImg + img #* ImageWeight
    return MergeImg