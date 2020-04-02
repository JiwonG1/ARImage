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
from scipy.optimize import curve_fit
from skimage.measure import block_reduce
# Import utils
import ibw_utils as ibw
import Color

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
    
    height, width = ref_img.shape if len(ref_img) == 2 else ref_img.shape[:2]
    if mannual:
        return (trans[0]/height, trans[1]/width), angle_define
    
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

    if count == 0:
        angle = angle_define

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

def auto_canny(image, sigma=0.33):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

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

def _Cosfunc(x, v, theta):
    return v * np.cos((x-theta)*np.pi/180)

def Vector(h5Files):
    
    x_data = [h5.angle for h5 in h5Files]
    y_dataStack = [np.rot90(h5.Piezoresponse(15, 13, fix=True),3) for h5 in h5Files]
    y_dataStack = np.stack(y_dataStack, axis=-1)
    sx, sy, _ = y_dataStack.shape 
    
    ## Apply Kernel 3*3 pixel, average pooling
    Xkernel_size = 3
    Ykernel_size = 3
    Xwindow = int((Xkernel_size-1)/2)
    Ywindow = int((Ykernel_size-1)/2)
    kernel_sx = divmod(sx, Xkernel_size)[0]
    kernel_sy = divmod(sy, Ykernel_size)[0]
    VectorOrigins = [(x, y) for x in range(1, kernel_sx+1) for y in range(1, kernel_sy+1)]
    QuiverOriginsX, QuiverOriginsY = [], [] #np.meshgrid([x for x in range(1, sx+1)], [y for y in range(1, sy+1)])
    QuiverU, QuiverV = [], []
    #VectorDict = {}
    print('Calculating Vectors')
    for x, y in tqdm(VectorOrigins):
        QuiverOriginsX.append(x*Xkernel_size)
        QuiverOriginsY.append(y*Ykernel_size)
        coord = (x*Xkernel_size, y*Ykernel_size)
        average_data = y_dataStack[coord[0]-Xwindow-1:coord[0]+Xwindow,coord[1]-Ywindow-1:coord[1]+Ywindow,:]
        average_data = average_data.sum(axis=0)
        average_data = average_data.sum(axis=0)
        #VectorDict[coord] = curve_fit(_Cosfunc, x_data, average_data, bounds=((0, -np.inf), (np.inf, np.inf)))[0]
        amp, phaseshift = curve_fit(_Cosfunc, x_data, average_data, bounds=((0, -np.inf), (np.inf, np.inf)))[0]
        QuiverU.append(amp*np.cos(phaseshift*np.pi/180))
        QuiverV.append(amp*np.sin(phaseshift*np.pi/180))
    plt.quiver(QuiverOriginsX, QuiverOriginsY, 2*QuiverU, 2*QuiverV, color='red')
    plt.savefig('./quivertest.png')
    plt.show()
    asdf
    #for key, val in VectorDict.items():
    #prnt(VectorDict)
    return y_dataStack[3]

def main(args):
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
    result_path = os.path.join(path, 'Results')
    mkfolder(result_path)
    ans = input('Choose image channel number for alignment\n')
    try:
        # If Flatten height image channel is exist,
        AlignChannel = int(ans)
        h5Files[0].ChannelName(AlignChannel)
    except:
        # Set alignment channel to Heightretrace
        print('Input channel has some problem, proceed with Heightretrace channel')
        AlignChannel = 2

    MidObj.alignimg = ibw.h5toimg(MidObj.data, AlignChannel)[0]
    MidObj.SaveBinaryImg(ibw.h5toimg(MidObj.data, TargetChannel)[0])
    
    print('\nCalculating...')
    
    for i in tqdm(range(len(h5Files))):
        h5file = h5Files[i]
        h5file.alignimg = ibw.h5toimg(h5file.data, AlignChannel)[0]
        if h5file.namedangle == 0:
            h5file.Piezoresponse(15, 13)
            continue
        else:
            h5file.trans, h5file.angle = alignment(h5file.alignimg, MidObj.alignimg, h5file.namedangle, args.mannual, score_threshold=0.9)
        ChannelImg = ibw.h5toimg(h5file.data, TargetChannel)[0]
        ChannelImg = imutils.translate(ChannelImg, h5file.trans[0], h5file.trans[1])
        ChannelImg = imutils.rotate(ChannelImg, h5file.angle)

        h5file.SaveBinaryImg(ChannelImg)
        h5file.Piezoresponse(15, 13, fix=True)

        h5Files[i] = h5file
    Vector(h5Files)
    #print('Result images are saved!')
    return 0

    ## TODO: Monte-Carlo method in paper to calculate alignment accurately
    ## Verctor with 3X3 kernel

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='Neural Network Training for Image Segmentation')
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('channel', help='Channel number to apply Angle Resolving')
    parser.add_argument('-m', '--mannual', action='store_true', help='Apply mannual angle tilting')
    args = parser.parse_args()

    main(args)