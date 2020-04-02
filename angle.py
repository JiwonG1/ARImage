# Import general functions
import argparse
import os, sys
from tqdm import tqdm
from pyprnt import prnt
import cv2
import math
import imutils
import numba
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters

# Import utils
import ibw_utils as ibw
import Color


data_dict = {
    3 : 'AmplitudeRetrace',
    5 : 'PhaseRetrace',
    13 : 'LatAmplitudeRetrace',
    15 : 'LatPhaseRetrace',
    16 : 'HeightRetrace'
}

"""class AngleImage(object):
    
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
        return ibw.channel_list(self.data)

    def ChannelName(self, channel):
        return ibw.h5toChannelName(self.data, channel)

    def Piezoresponse(self, phasechannel, ampchannel, fix=False):
        #binarizedphase = self.SaveBinaryImg(ibw.h5toimg(self.data, phasechannel)[0], response=True)
        phaseimg = ibw.h5toimg(self.data, phasechannel)[0]
        ampimg = ibw.h5toimg(self.data, ampchannel)[0]
        piezoresponse = ampimg * np.cos(phaseimg*np.pi)
        if fix:
            piezoresponse = imutils.translate(piezoresponse, self.trans[0], self.trans[1])
            piezoresponse = imutils.rotate(piezoresponse, self.angle)
        #self.piezo = piezoresponse.astype(np.uint8)
        self.piezo = piezoresponse
        return self.piezo
    
    def NormPiezoresponse(self, phasechannel, ampchannel, fix=False):
        piezo = self.Piezoresponse(phasechannel, ampchannel)
        Normpiezo = np.zeros(piezo.shape)
        Normpiezo = cv2.normalize(piezo, Normpiezo, 0, 255, cv2.NORM_MINMAX)
        if fix:
            Normpiezo = imutils.translate(Normpiezo, self.trans[0], self.trans[1])
            Normpiezo = imutils.rotate(Normpiezo, self.angle-90)
        self.Normpiezo = Normpiezo
        return self.Normpiezo"""
        
### Not Using
def Mapping(objlist, StartColorNumber, interval):
    obj3D = objlist[0].piezo
    
    for i in range(len(objlist[1:])):
        obj3D = np.dstack((obj3D, objlist[i+1].Normpiezo))
    obj3Dsign = np.sign(obj3D)
    signchange = ((obj3Dsign - np.roll(obj3Dsign, 1, axis=2)) != 0).astype(int)
    signchange = signchange[:, :, :-1]
    #ColorCode = np.argmax(signchange)
    #ColorCode = np.argmin(np.abs(obj3D),axis=2)
    ImageStack = Color.Coloring(signchange[:, :, 0], Color.ColorCode(Color.ColorOrder(1)))
    for j in range(signchange.shape[2]-1):
        ImageStack = np.dstack((ImageStack, Color.Coloring(signchange[:, :, j+1], Color.ColorCode(Color.ColorOrder(j+2)))))
        #Image = signchange[:, :, j] * Color.ColorCode(Color.ColorOrder(j))
    Imageshape = signchange[:, :, 0].shape
    #Image = np.zeros((Imageshape[0], Imageshape[1], 3))
    #for k in range(ImageStack.shape[2]):
    #    if k % 3 == 0:   ##RED
    #        Image[:,:,0] += ImageStack[:,:,k]
    #    elif k % 3 == 1: ##GREEN
    #        Image[:,:,1] += ImageStack[:,:,k]
    #    else:            ##BLUE
    #        Image[:,:,2] += ImageStack[:,:,k]
    #np.repeat(a[:, :, np.newaxis], 3, axis=2)
    #numerator = np.sum(ImageStack, axis=2)
    #denominator = np.repeat(np.sum(signchange, axis=2)[:,:,np.newaxis], 3, axis=2)
    #MergeImg = np.divide(Image, denominator, out=np.zeros_like(Image), where=denominator!=0) 
    ## RGB to RGBA (alpha value) Check 
    return signchange#ImageStack[:,:,:3]#MergeImg

def alignment(img1, ref_img, angle_define, mannual=False, score_threshold=0.8):
    
    MAX_FEATURE = 500
    GOOD_MATCH_PERCENT = 0.10

    if len(ref_img.shape) > 2:
        img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else: 
        img1gray, ref_img_gray = img1, ref_img
    # Edge image test
    img1gray = auto_canny(img1gray)
    ref_img_gray = auto_canny(ref_img_gray)
    orb = cv2.ORB_create(MAX_FEATURE)
    keypoint1, descriptors1 = orb.detectAndCompute(img1gray, None)
    keypoint2, descriptors2 = orb.detectAndCompute(ref_img_gray, None)
    
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors1, descriptors2, None)

    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
   
    imMatches = cv2.drawMatches(img1, keypoint1, ref_img, keypoint2, matches, None)
    imMatches_edge = cv2.drawMatches(img1gray, keypoint1, ref_img_gray, keypoint2, matches, None)
    cv2.imwrite('./data/matches_'+str(angle_define)+'.jpg', imMatches)
    cv2.imwrite('./data/edgematches_'+str(angle_define)+'.jpg', imMatches_edge)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i,:] = keypoint1[match.queryIdx].pt
        points2[i,:] = keypoint2[match.trainIdx].pt 
    
    h= cv2.findHomography(points1, points2, cv2.RANSAC)[0]
    h = h[:2]
    trans = decompose_homography(h)[0]
    
    #print(trans, angle, scale, shear)
    height, width = ref_img.shape if len(ref_img) == 2 else ref_img.shape[:2]
    if mannual:
        return (trans[0]/height, trans[1]/width), angle_define
    
    #img1 = cv2.warpAffine(img1, translate, (height, width))
    img1 = imutils.translate(img1, trans[0]/height, trans[1]/width)
    ref_img_edge = auto_canny(ref_img_gray)
    #score_threshold = 0.8
    count = 0
    for a in np.arange(angle_define-5,angle_define+5, 0.1):
        img1Reg = imutils.rotate(img1, a)
        img1Reg_edge = auto_canny(img1Reg)
        score = CompareEdge(img1Reg_edge, ref_img_edge, threshold=5, margin=100)
       
        if score > score_threshold:
            score_threshold = score
            angle = a
            count += 1
            #print('angle is updated!')
            #img1Reg_edge_high = img1Reg_edge
    if count == 0:
        angle = angle_define

    #imgReg = imutils.rotate(img1, angle)
    #overlay_img = overlay(imgReg, ref_img, 0.5)
    #cv2.imwrite('./data/'+str(angle_define)+'_overlay.jpg', overlay_img)
    #cv2.imwrite('./edge_ref.jpg', ref_img_edge)
    #cv2.imwrite('./edge_im.jpg', img1Reg_edge_high)
    #print(np.round(angle, 3))
    return (trans[0]/height, trans[1]/width), angle

def mkfolder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def decompose_homography(mat):
    a = mat[0,0]
    b = mat[0,1]
    c = mat[0,2]
    d = mat[1,0]
    e = mat[1,1]
    f = mat[1,2]

    p = math.sqrt(a**2 + b**2)
    r = (a*e - b*d)/p
    q = (a*d + b*e)/(a*e - b*d)

    trans = (c, f)
    scale = (p, r)
    shear = q
    theta = math.atan2(b,a) * 180 / math.pi

    return trans, theta, scale, shear

### Not Using
def overlay(img, img_ref, alpha=0.5):

    img, img_ref = img.astype(np.uint8), img_ref.astype(np.uint8)

    if len(img.shape) == 2:
        img = np.stack([img for _ in range(3)], axis=2)
    if len(img_ref.shape) == 2:
        img_ref = np.stack([img_ref for _ in range(3)], axis=2)
    
    beta = 1.0 - alpha
    
    overlay = cv2.addWeighted(img, alpha, img_ref, beta, 0.0)

    return overlay

def auto_canny(image, sigma=0.33):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

### Not Using
def binary_img(img, trans, angle, cut=False, invert=False):
    height, width = img.shape if len(img.shape) == 2 else img.shape[:2]
    #img = (img / (np.max(img) - np.min(img))*256).astype(np.uint8)
    img = imutils.translate(img, trans[0], trans[1])
    img = imutils.rotate(img, angle)
    if cut:
        h, w = int(height * (0.5/np.sqrt(2))), int(width * (0.5/np.sqrt(2)))
        img = img[int(height/2-h):int(height/2+h),int(width/2-w):int(width/2+w)]
    if invert:
        _, img_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        _, img_result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #plt.imsave('test_cov.png', img_result, cmap='gray')
    #print(np.unique(img_result))
    return img_result
### Not Using
def domain(img, img_ref):
    
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

# Compute precistion and recall given contours
@numba.njit
def calc_precision_recall(contours_a, contours_b, threshold):

    count = 0
    for b in range(len(contours_b)):
        # find the nearest distance
        for a in range(len(contours_a)):
            distance = (contours_a[a][0]-contours_b[b][0])**2 + (contours_a[a][1]-contours_b[b][1]) **2

            #distance = numba.literally(distance)
            #if distance(contours_a[a][0],contours_b[b][0],contours_a[a][1],contours_b[b][1]) < threshold:
            #print()
            if distance < threshold **2:
                count = count + 1
                break

    if count != 0:
        precision_recall = count/len(contours_b)
    else:
        precision_recall = 0

    return precision_recall, count, len(contours_b)

# Compute distance between two points, using numba decorator
@numba.njit
def distance(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def CompareEdge(edge, ref_edge, threshold=5, margin=256):

    x_mid, y_mid = ref_edge.shape[0]/2, ref_edge.shape[1]/2
    tmp = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    edge_contours = tmp[0] if len(tmp) == 2 else tmp[1]
    edge_contour = [edge_contours[i][j][0].tolist() for i in range(len(edge_contours)) for j in range(len(edge_contours[i]))]
    edge_contour = [coord for coord in edge_contour if 
                    abs(coord[0]-x_mid) < margin and abs(coord[1] - y_mid) < margin]
    tmp = cv2.findContours(ref_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ref_contours = tmp[0] if len(tmp) == 2 else tmp[1]
    ref_contour = [ref_contours[i][j][0].tolist() for i in range(len(ref_contours)) for j in range(len(ref_contours[i]))]
    ref_contour = [coord for coord in ref_contour if 
                    abs(coord[0]-x_mid) < margin and abs(coord[1] - y_mid) < margin]

    #ref_contour = numba.types.literal(np.int_(np.array(ref_contour)))
    #edge_contour = numba.types.literal(np.int_(np.array(edge_contour)))
    ref_contour = np.int_(np.array(ref_contour))
    edge_contour = np.int_(np.array(edge_contour))

    precision = calc_precision_recall(
            ref_contour, edge_contour, np.array(threshold))[0]    # Precision
        #print("\tprecision:", denominator, numerator)

    recall= calc_precision_recall(
        edge_contour, ref_contour, np.array(threshold))[0]    # Recall
    #print("\trecall:", denominator, numerator)
    
    #print("\trecall:", denominator, numerator)
    if precision == 0 or recall == 0:
        f1 = np.nan
    else:
        f1 = 2*recall*precision/(recall+precision)

    return f1

def AngleResolved(args):
    path = os.path.join('./data',args.dataset)
    h5Files = []
    print('\n \tLoading the ibw files...')

    for filename in tqdm([f for f in os.listdir(path) if f[-3:] == 'ibw']):
        if os.path.isfile(os.path.join(path, filename[:-3]+'h5')):
            h5 = ibw.ibw2h5(os.path.join(path, filename[:-3]+'h5'), exist=True)
        else:
            h5 = ibw.ibw2h5(os.path.join(path, filename))
        ImageObj = ibw.AngleImage(filename[:-4], h5)
        h5Files.append(ImageObj)
        if filename.split('_')[-1][:-4] == str(0):
            MidObj = ImageObj

    #MidObj = [obj for obj in h5Files if obj.angle == 0][0]
    h5Files = sorted(h5Files, key=lambda h5: h5.namedangle)
    angles = [obj.namedangle for obj in h5Files]
    angle_diff = list(set([j-i for i, j in zip(angles[:-1], angles[1:])]))

    if not len(angle_diff) == 1:
        print('Angle differences between images are not even!')
        ans = input('Which angle do you want to use for angle-resolving?')
        angle_diff = int(ans)
    else:
        angle_diff = angle_diff[0]
    print('\nDefined Angle between images is {}'.format(angle_diff))
    
    ## Channel Info
    TypeNum, channels = ibw.channel_list_folder(h5Files)
    if TypeNum == 1:
        print('\n Channel list')
        prnt(channels)
    else:
        type_dict, type_files, channel_type = channels
        print('\nThere are multiple types of files in the folder...')
        for key, val in type_dict.items():
            print('Type {}'.format(key))
            print('{} files in this type:\n{}'.format(len(val.split(',')), val))
            print('Channel List of Type {}'.format(key))
            prnt(channel_type[type_files[key-1]])
    channel_name = h5Files[0].ChannelName(args.channel)
    ans = input('Selected channel is [Channel_{0}:{1}]. \nDo you want to proceed?\n'.format(args.channel, channel_name))
    ans = ans.lower()
    if ans == 'y' or ans == 'ye' or ans == 'yes' or ans == '':
        TargetChannel = args.channel
    else:
        ans = input('\nPlease select channel Number or type any letter for exit.\n')
        try:
            TargetChannel = int(ans)
            channel_name = h5Files[0].ChannelName(TargetChannel)
            print('Selectd channel is [Channel_{0}:{1}].'.format(TargetChannel, channel_name))
        except:
            print('Aborted!')
            sys.exit(1)
    ## 시계 방향이 + 방향

    # Result information
    #channel_name = h5Files[0].ChannelName(TargetChannel)
    #ibw.h5toChannelName(h5Files[0].data, args.channel)
    result_path = os.path.join(path, 'Results')
    mkfolder(result_path)
    ans = input('Choose image channel number for alignment\n')
    try:
        # If Flatten height image channel is exist,
        #ibw.h5toChannelName(h5Files[0].data, 1)
        AlignChannel = int(ans)
        h5Files[0].ChannelName(AlignChannel)
        #HeightChannel = 16
    except:
        # Set alignment channel to Heightretrace
        print('Input channel has some problem, proceed with Heightretrace channel')
        AlignChannel = 2

    MidObj.alignimg = ibw.h5toimg(MidObj.data, AlignChannel)[0]
    MidObj.SaveBinaryImg(ibw.h5toimg(MidObj.data, TargetChannel)[0])
    #MidObj.NormPiezoresponse(15, 13)
    #overlay_show = MidObj.heightimg
    ## Clockwise
    #Clockwise = [obj for obj in h5Files if obj.angle > 0]
    #Clockwise[0] = MidObj
    print('\nCalculating...')
    #Clockwise_tilt = {}
    tilting = 1
    for i in tqdm(range(len(h5Files))):
        h5file = h5Files[i]
        h5file.alignimg = ibw.h5toimg(h5file.data, AlignChannel)[0]
        #print(h5file.namedangle)
        if h5file.namedangle == 0:
            #print('pass!')
            h5file.Piezoresponse(15, 13)
            continue
        else:
        #elif h5file.namedangle > 0:
            h5file.trans, h5file.angle = alignment(h5file.alignimg, MidObj.alignimg, h5file.namedangle, args.mannual, score_threshold=0.9)
        #else:
            #h5file.trans, h5file.angle = alignment(h5file.heightimg, MidObj.heightimg, angle_diff*tilting, args.mannual, score_threshold=0.9)
        #print(h5file.angle)
        #fixed_height = imutils.rotate(imutils.translate(h5file.heightimg, h5file.trans[0], h5file.trans[1]), h5file.angle)
        #fixed_height = cv2.cvtColor(overlay(MidObj.alignimg, fixed_height), cv2.COLOR_BGR2GRAY)
        #overlay_show = np.hstack([overlay_show, fixed_height])
        ChannelImg = ibw.h5toimg(h5file.data, TargetChannel)[0]
        ChannelImg = imutils.translate(ChannelImg, h5file.trans[0], h5file.trans[1])
        ChannelImg = imutils.rotate(ChannelImg, h5file.angle)

        h5file.SaveBinaryImg(ChannelImg) # BinaryImg = 
        h5file.Piezoresponse(15, 13, fix=True)
        #h5file.NormPiezoresponse(15, 13, fix=True) # BinaryImg = 

        h5Files[i] = h5file
        tilting += 1
        #print(i)
    
    TurnWhite, TurnBlack = Color.ColorMapping(h5Files, 1)
    Result = Color.Blending(TurnWhite+TurnBlack)
    Result_adjust = Color.AdjustGamma(Result, 4)
    cv2.imwrite(result_path+'/ARPFM_{}.png'.format(channel_name), Result)
    cv2.imwrite(result_path+'/ARPFM_{}_G.png'.format(channel_name), Result_adjust)
    print('Result images are saved!')
    img = ibw.AngleMap(h5Files)
    cv2.imwrite('./test_stacking.jpg', img)
    return 0

    

    ## TODO: Monte-Carlo method in paper to calculate alignment accurately
    ## Verctor with 3X3 kernel


    ### Below is Not Using 
    """
    #BlendingImg1 = np.hstack([BlendingImg, adjust_g])
    #BlendingImg2 = Color.Blending(TurnBlack)
    #cv2.imwrite('./overlay_ex2.jpg', overlay_show)
    #cv2.imwrite('./data/turnwhite_test2.jpg', BlendingImg)
    cv2.imwrite('./data/Brighter3.jpg', Result_adjust)
    
    plt.imsave('./data/turnblack_test2.png', np.rollaxis(BlendingImg2,0,2)/255)
    cv2.imwrite('./data/binary1.jpg', h5Files[0].binaryimg)
    cv2.imwrite('./data/binary2.jpg', h5Files[1].binaryimg)
    #for obj1, obj2 in zip(h5Files[:-1], h5Files[1:]):
        #turnwhite, turnblack = Compare(obj1.binaryimg, obj2.binaryimg)

    adsf
    for i in range(len(Clockwise)):
        
        h5obj = Clockwise[i]
        h5obj.heightimg = ibw.h5toimg(h5obj.data, HeightChannel)[0]
        h5obj.trans, h5obj.angle = alignment(h5obj.heightimg, MidObj.heightimg, angle_diff*tilting, args.mannual, score_threshold=0.90)
        ObjChannelImg = ibw.h5toimg(h5obj.data, args.channel)[0]
        ObjChannelImg = imutils.translate(ObjChannelImg, h5obj.trans[0], h5obj.trans[1])
        ObjChannelImg = imutils.rotate(ObjChannelImg, h5obj.angle)

        binaryimg = h5obj.SaveBinaryImg(ObjChannelImg)
        #cv2.imwrite(result_path+'/binary_{}.jpg'.format(h5obj.imagename), binaryimg)

        piezo = h5obj.NormPiezoresponse(15, 13, fix=True)
        #cv2.imwrite(result_path+'/piezo_{}.jpg'.format(h5obj.imagename), piezo)

        Clockwise[i] = h5obj
        print('Complete calculation for "{}"'.format(h5obj.imagename))
        tilting += 1
    #ClockWhite, ClockBlack = ColorMapping(Clockwise, 1)
    #TestList = ClockWhite + ClockBlack
    #Merged = MergeImg(TestList)
    #print(Merged)
    #cv2.imwrite('./data/merge_test.jpg', Merged)
    ColorNumber = 1
    for data1, data2 in zip(Clockwise[:-1], Clockwise[1:]):
        turnwhite, turnblack = domain(data1.binaryimg, data2.binaryimg)

        WhiteImg = Color.Coloring(turnwhite, Color.ColorCode(Color.ColorOrder(ColorNumber)))
        BlackImg = Color.Coloring(turnblack, Color.ColorCode(Color.ColorOrder(ColorNumber+12)))
        #cv2.imwrite(result_path+'/Binary_{}.jpg'.format(ColorNumber*angle_diff),data1.binaryimg)
        #cv2.imwrite(result_path+'/Gray_{}.jpg'.format(ColorNumber*angle_diff),data1.imgdata)

        cv2.imwrite(result_path+'/AR_{}.jpg'.format(ColorNumber*angle_diff),WhiteImg)
        cv2.imwrite(result_path+'/AR_B_{}.jpg'.format(ColorNumber*angle_diff),BlackImg)
        ColorNumber += 1
        
    # CounterClockwise
    CounterClockwise = sorted([obj for obj in h5Files if obj.angle < 0], key=lambda obj: obj.angle, reverse=True)
    #CounterClockwise[0] = MidObj
    print('\nCalculating Counter Clockwise direction...')
    tilting = -1
    for i in range(len(CounterClockwise)):
        
        h5obj = CounterClockwise[i]
        ObjImg = ibw.h5toimg(h5obj.data, HeightChannel)[0]
        trans, angle = alignment(ObjImg, MidImg, angle_diff*tilting, args.mannual, score_threshold=0.90)
        h5obj.InputCalculated(trans, angle)
        
        ObjChannelImg = ibw.h5toimg(h5obj.data, args.channel)[0]
        ObjChannelImg = imutils.translate(ObjChannelImg, h5obj.trans[0], h5obj.trans[1])
        ObjChannelImg = imutils.rotate(ObjChannelImg, h5obj.angle)

        h5obj.SaveImg(ObjImg)
        h5obj.SaveBinaryImg(ObjChannelImg)
        h5obj.NormPiezoresponse(15, 13, fix=True)

        CounterClockwise[i] = h5obj

        print('Complete calculation for "{}"'.format(h5obj.imagename))
        tilting -= 1
    #CounterClockWhite, CounterClockBlack = ColorMapping(CounterClockwise, -1)
    h5Files_cal = CounterClockwise[::-1] + [MidObj] +Clockwise
    #print(h5Files_cal)
    obj3D = Mapping(h5Files_cal, 1, 1)
    
    print(obj3D)
    print(obj3D.shape)
    print(obj3D[250, 0, :])
    print(obj3D[100, 256, :])
    print(obj3D[0, 118, :])
    #cv2.imwrite(result_path+'/mergetest.jpg', obj3D)
    #plt.imsave('./data/plttest.png', obj3D/255)
    #print(obj3D[0,0,:])
    #print(np.unique(np.sum(obj3D, axis=2)))
    asdfs
    Testlist = ClockWhite + CounterClockWhite
    Testlist2 = ClockBlack + CounterClockBlack
    Testlist3 = Testlist + Testlist2
    Merged = MergeImg(Testlist)
    Merged2 = MergeImg(Testlist2)
    Merged3 = MergeImg(Testlist3)
    cv2.imshow(Merged2)
    cv2.imwrite('./data/merge_test.jpg', Merged)
    cv2.imwrite('./data/merge_test2.jpg', Merged2)
    cv2.imwrite('./data/merge_test3.jpg', Merged3)
    #for data1, data2 in tqdm(zip(CounterClockwise[:-1], CounterClockwise[1:])):
        #CounterClockwise_result.append(CompareImageCounter(data1.data, data2.data,
        #                        angle_diff, args.channel, tilting, MidObj[0].data, HeightChannel=16))
        #tilting -= 1
    #print(CounterClockwise_result)
    ColorNumber = -1
    for data1, data2 in zip(CounterClockwise[:-1], CounterClockwise[1:]):
        turnwhite, turnblack = domain(data1.binaryimg, data2.binaryimg)

        WhiteImg = Color.Coloring(turnwhite, Color.ColorCode(Color.ColorOrder(ColorNumber)))
        BlackImg = Color.Coloring(turnblack, Color.ColorCode(Color.ColorOrder(ColorNumber+12)))
        #cv2.imwrite(result_path+'/Binary_{}.jpg'.format(ColorNumber*angle_diff),data1.binaryimg)
        #cv2.imwrite(result_path+'/Gray_{}.jpg'.format(ColorNumber*angle_diff),data1.imgdata)

        cv2.imwrite(result_path+'/ARC_{}.jpg'.format(ColorNumber*angle_diff),WhiteImg)
        cv2.imwrite(result_path+'/ARC_B_{}.jpg'.format(ColorNumber*angle_diff),BlackImg)
        ColorNumber -= 1
    
    #for ColorNumber, images in enumerate(CounterClockwise_result, start=0):
    #    turnwhite, turnblack = images
    #    ColorNumber = - ColorNumber 
    #    WhiteImg = Color.Coloring(turnwhite, Color.ColorCode(Color.ColorOrder(ColorNumber)))
    #    BlackImg = Color.Coloring(turnblack, Color.ColorCode(Color.ColorOrder(ColorNumber+12)))

    #    cv2.imwrite(result_path+'/ARC_{}.jpg'.format(ColorNumber*angle_diff),WhiteImg)
    #    cv2.imwrite(result_path+'/ARC_B_{}.jpg'.format(ColorNumber*angle_diff),BlackImg)

    
    """

### Not Using
def CompareImage(data1, data2, angle_define, channel, tilting, Zero,
                 HeightChannel=1, cut=False, invert=False):
    img0 = ibw.h5toimg(Zero, HeightChannel)[0]
    img1 = ibw.h5toimg(data1, HeightChannel)[0]
    img2 = ibw.h5toimg(data2, HeightChannel)[0]

    trans, angle = alignment(img2, img1, angle_define)

    tilting1 = alignment(img1, img0, tilting)[1]
    tilting2 = alignment(img2, img0, tilting)[1]

    ApplyImg1 = ibw.h5toimg(data1, channel)[0]
    ApplyImg2 = ibw.h5toimg(data2, channel)[0]
    
    #BWImg1 = binary_img(ApplyImg1, trans, tilting, cut, invert)
    #BWImg2 = binary_img(ApplyImg2, trans, angle+tilting, cut, invert)
    BWImg1 = binary_img(ApplyImg1, trans, tilting1, cut, invert)
    BWImg2 = binary_img(ApplyImg2, trans, angle+tilting2, cut, invert)

    TurnWhite, TurnBlack = domain(BWImg2, BWImg1)

    return (TurnWhite, TurnBlack)
### Not Using
def CompareImageCounter(data1, data2, angle_define, channel, tilting, Zero,
                 HeightChannel=1, cut=False, invert=False):
    img0 = ibw.h5toimg(Zero, HeightChannel)[0]
    img1 = ibw.h5toimg(data1, HeightChannel)[0]
    img2 = ibw.h5toimg(data2, HeightChannel)[0]

    trans, angle = alignment(img2, img1, angle_define)
    tilting1 = alignment(img1, img0, tilting)[1]
    tilting2 = alignment(img2, img0, tilting)[1]

    ApplyImg1 = ibw.h5toimg(data1, channel)[0]
    ApplyImg2 = ibw.h5toimg(data2, channel)[0]
    
    #BWImg1 = binary_img(ApplyImg1, trans, angle+tilting, cut, invert)
    #BWImg2 = binary_img(ApplyImg2, trans, tilting, cut, invert)
    BWImg1 = binary_img(ApplyImg1, trans, angle+tilting1, cut, invert)
    BWImg2 = binary_img(ApplyImg2, trans, tilting2, cut, invert)

    TurnWhite, TurnBlack = domain(BWImg2, BWImg1)

    return (TurnWhite, TurnBlack)

if __name__ == '__main__':
 
    """ibw_path = './data/pvdf/'
    channel_num = 16
    h5_ref = ibw.ibw2h5(ibw_path+'0.ibw')
    #im_ref = ibw.ibw2img(ibw_path+'0.ibw', 16)[0]
    im_ref, cmap_ref= ibw.h5toimg(h5_ref, channel_num)
    h5_img = ibw.ibw2h5(ibw_path+'15.ibw')
    #im = ibw.ibw2img(ibw_path+'15.ibw', 16)[0]
    im, cmap_im = ibw.h5toimg(h5_img, channel_num)
   
    #plt.imsave(ibw_path+'0.png', im_ref, cmap=cmap_ref)
    #plt.imsave(ibw_path+'15.png', im, cmap=cmap_im)

    #ref_file = ibw_path+'0.png'
    #imref = cv2.imread(ref_file, cv2.IMREAD_COLOR)
    #angle_file = ibw_path+'15.png'
    #im = cv2.imread(angle_file, cv2.IMREAD_COLOR)

    #trans, angle = alignment(im, imref, 15)
    trans, angle = alignment(im, im_ref, 15)
    #print(trans, angle)
    apply_num = 15
    #ch_name, ch_img = ibw.h5toChannelImg(h5_img, apply_num)
    #ch_img_ref = ibw.h5toChannelImg(h5_ref, apply_num)[1]
    apply_img = ibw.h5toimg(h5_img, apply_num)[0]
    apply_ref = ibw.h5toimg(h5_ref, apply_num)[0]
    ch_name = ibw.h5toChannelName(h5_img, apply_num)
    print(ch_name)
    bw_img = binary_img(apply_img, trans, angle, cut=True)
    bw_img_ref = binary_img(apply_ref, (0, 0), 0, cut=True, invert=False)
    plt.imsave('bw_im_ref.png',bw_img_ref, cmap='gray')
    plt.imsave('bw_im.png',bw_img, cmap='gray')

    turn_w, turn_b = domain(bw_img, bw_img_ref)
    white = Color.Coloring(turn_w, Color.ColorCode(Color.ColorOrder(1)))
    black = Color.Coloring(turn_b, Color.ColorCode(Color.ColorOrder(13)))

    overlay_img1 = overlay(white*255, bw_img_ref)
    overlay_img2 = overlay(black*255, bw_img_ref)
    #white = np.zeros(turn_w.shape)
    #black = np.zeros(turn_b.shape)
    #white[turn_w] = 255
    #black[turn_b] = 255

    #outfile = 'test.jpg'
    #cv2.imwrite(outfile, imReg)
    plt.imsave('turn_white.png',white)
    plt.imsave('turn_black.png',black)
    plt.imsave('overlay_white.png',overlay_img1)
    plt.imsave('overlay_black.png',overlay_img2)
    
    #plt.imsave(ibw_path+ch_name+'_0.png',ch_img, cmap='gray')"""

    parser = argparse.ArgumentParser(description='Neural Network Training for Image Segmentation')
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('channel', help='Channel number to apply Angle Resolving')
    parser.add_argument('-m', '--mannual', action='store_true', help='Apply mannual angle tilting')
    args = parser.parse_args()
    AngleResolved(args)