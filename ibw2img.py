import os, sys, argparse
import h5py
import ibw_utils as ibw
from tqdm import tqdm
from pyprnt import prnt
from matplotlib import pyplot as plt

def main(args):
    path = os.path.join('./data',args.dataset)
    h5Files = []
    print('Loading the ibw files...')

    for filename in tqdm([f for f in os.listdir(path) if f[-3:] == 'ibw']):
        if os.path.isfile(os.path.join(path, filename[:-3]+'h5')):
            h5 = h5py.File(os.path.join(path, filename[:-3]+'h5'), 'r')
            h5 = h5['Measurement_000/']
        else:
            h5 = ibw.ibw2h5(os.path.join(path, filename))
        ImageObj = ibw.AngleImage(filename[:-4], h5)
        h5Files.append(ImageObj)
    h5Files = sorted(h5Files, key=lambda h5: h5.namedangle)

    if args.channel == str(999):
        TypeNum, channels = ibw.channel_list_folder(h5Files)
        if TypeNum == 1:
            print('\nChannel List of {} Folder'.format(args.dataset))
            prnt(channels)
        else:
            type_dict, type_files, channel_type = channels
            print('\nThere are multiple types of files in the folder...')
            for key, val in type_dict.items():
                print('Type {}'.format(key))
                print('{} files in this type:\n{}'.format(len(val.split(',')), val))
                print('Channel List of Type {}'.format(key))
                prnt(channel_type[type_files[key-1]])
    elif args.channel == str(998):
        channels = ibw.channel_list(h5Files[0].data)
        ampchannel = input('Please enter the amplitude channel number:\n')
        try:
            ampchannel = int(ampchannel)
        except:
            print('Please enter channel number as integer!! \n Exiting!')
            sys.exit(1)
        phasechannel = input('Plaes enter the phase channel number:\n')
        try:
            phasechannel = int(phasechannel)
        except:
            print('Please enter channel number as integer!! \n Exiting!')
            sys.exit(1)
        PhaseChannelName = channels['Channel_'+str(phasechannel).zfill(3)]
        AmpChannelName = channels['Channel_'+str(ampchannel).zfill(3)]

        ImagePath = os.path.join(path, 'Images')
        if PhaseChannelName[:3] == 'Lat' and AmpChannelName[:3] == 'Lat':
            if args.norm:
                ChannelPath = os.path.join(ImagePath, 'LatPiezoNorm')
            else:
                ChannelPath = os.path.join(ImagePath, 'LatPiezo')
            print('Selected channel is Lateral Piezoresponse images.')
        elif PhaseChannelName[:5] == 'Phase' and AmpChannelName[:9] == 'Amplitude':
            if args.norm:
                ChannelPath = os.path.join(ImagePath, 'VerPiezoNorm')
            else:
                ChannelPath = os.path.join(ImagePath, 'VerPiezo')
            print('Selected channel is Vertical Piezoresponse images.')
        else:
            print('Please match the channel type! \n Exiting!')
            sys.exit(1)
        mkfolder(ImagePath)
        mkfolder(ChannelPath)

        for idx in range(len(h5Files)):
            if args.norm:
                PiezoImg = h5Files[idx].NormPiezoresponse(phasechannel, ampchannel)
            else:
                PiezoImg = h5Files[idx].Piezoresponse(phasechannel, ampchannel)
            plt.imsave(os.path.join(ChannelPath, h5Files[idx].imagename+'.png'), PiezoImg, cmap='gray')
            print(h5Files[idx].imagename+'.png', 'is saved!')
    else:
        channels = ibw.channel_list(h5Files[0].data)
        if not args.channel:
            print('You have to designate channel! \n')
            sys.exit(1)

        ChannelName = channels['Channel_'+str(args.channel).zfill(3)]
        print('Selected channel name: ', ChannelName)
        
        ImagePath = os.path.join(path, 'Images')
        if args.flatten:
            ChannelPath = os.path.join(ImagePath, ChannelName+'_Flatten')
        else:
            ChannelPath = os.path.join(ImagePath, ChannelName)    
        mkfolder(ImagePath)
        mkfolder(ChannelPath)

        for idx in range(len(h5Files)):
            img, cmap_img = ibw.h5toimg(h5Files[idx].data, str(args.channel))
            if args.flatten:
                img = ibw.ImageFlatten(img, int(args.flatten))
            plt.imsave(os.path.join(ChannelPath, h5Files[idx].imagename+'.png'), img, cmap=cmap_img)
            print(h5Files[idx].imagename+'.png', 'is saved!')

def mkfolder(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Network Training for Image Segmentation')
    parser.add_argument('dataset', help='name of the dataset folder')
    parser.add_argument('channel', help='Channel number of converting images')
    parser.add_argument('-n', '--norm', action='store_true', help='Apply normalization when generating piezoresponse images')
    parser.add_argument('-f', '--flatten', nargs='?', default=False, help='Apply flatten with order')
    args = parser.parse_args()
    main(args)