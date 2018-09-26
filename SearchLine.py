# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import os
import scipy.ndimage
import argparse
import os.path

'''

USAGE: "python SearchLine.py -h" will give a description of the input values

python SearchLine.py -Cube cube.fits -MaxSigmas 10 -MinSN 3.5 -OutputPath LineSearchTEST1 -UseMask True -ContinuumImage continuum.fits -MaskSN 5.0

Changelog:
---------------------------------------------------------------------------------------------
SearchLine_v0.py
Script that search emission lines features. 

---------------------------------------------------------------------------------------------
SearchLine_v0.1.py
Now it gives the option to use the continuum image to mask continuum sources.

---------------------------------------------------------------------------------------------
v0.2
Updated documentation and changed the naming convention where the version will be in the header.

---------------------------------------------------------------------------------------------
v0.3
Change to use fits files as the output instead of ascii.
'''

def GetValue(CubePath):
    hdulist =   fits.open(CubePath,memmap=True)
    head = hdulist[0].header
    data = hdulist[0].data[0]
    try:
        BMAJ = hdulist[1].data.field('BMAJ')
        BMIN = hdulist[1].data.field('BMIN')
        BPA = hdulist[1].data.field('BPA')
    except:
        BMAJ = []
        BMIN = []
        BPA = []
        for i in range(len(data)):
            BMAJ.append(head['BMAJ']*3600.0)
            BMIN.append(head['BMIN']*3600.0)
            BPA.append(head['BPA'])
        BMAJ = np.array(BMAJ)
        BMIN = np.array(BMIN)
        BPA = np.array(BPA)
    pix_size = head['CDELT2']*3600.0
    factor = 2*(np.pi*BMAJ*BMIN/(8.0*np.log(2)))/(pix_size**2)

    RefFrequency = head['CRVAL3']
    ChannelSpacing = head['CDELT3']
    ApproxChannelVelocityWidth = ( abs(ChannelSpacing)/RefFrequency ) * 3e5
    ApproxMaxSigmas = 1000.0/ApproxChannelVelocityWidth


    aux = len(data[0][np.isfinite(data[0])].flatten())*1.0/factor[0]*(len(data)/ApproxMaxSigmas)
    print '*** A rough guesstimate to use as MinSN is',round(np.power(10,np.log10(aux)*0.07723905 + 0.19291493),1),'***'
    # print len(data[0][np.isfinite(data[0])].flatten())*1.0/factor[0]*(len(data)/ApproxMaxSigmas)
    return 

def SearchLine(CubePath,FolderForLinesFiles,MinSN,sigmas,UseMask,ContinuumImage,MaskSN,Kernel):

    SN = np.array([])
    SNneg = np.array([])

    print 100*'#'
    print 'Starting search of lines with sigma equal to',sigmas,'channels'



    hdulist =   fits.open(CubePath,memmap=True)
    data  = hdulist[0].data[0]  
    hdulist = 0


    #Nop optimal, but it reads the continuum image every cycle.
    if UseMask:
        DataMask = fits.open(ContinuumImage,memmap=True)[0].data[0][0]  
        InitialRMS = np.nanstd(DataMask)
        FinalRMS = np.nanstd(DataMask[DataMask<MaskSN*InitialRMS])
        Mask = np.where(DataMask>=MaskSN*FinalRMS,True,False)

    if Kernel=='Gaussian' or Kernel=='guassian':
        data = scipy.ndimage.filters.gaussian_filter(data, [sigmas,0,0],mode='constant', cval=0.0, truncate=4.0)
    else:
        ZeroChannel = []
        NanChannel = []
        for ch in data:
            if len(ch[ch==0].flatten())*1.0>0.9*(len(ch.flatten())):
                ZeroChannel.append(True)
            else:
                ZeroChannel.append(False)
            if len(ch[np.isnan(ch)].flatten())*1.0>0.9*(len(ch.flatten())):
                NanChannel.append(True)
            else:
                NanChannel.append(False)

        ZeroChannel = np.array(ZeroChannel)
        NanChannel = np.array(NanChannel)
        data = scipy.ndimage.filters.uniform_filter(data, [sigmas,0,0],mode='mirror',cval=0.0)
        data[ZeroChannel] = 0
        data[NanChannel] = np.nan


    # newdata = []
    # for l in range(len(data)):
    #     aux = data[max(l - sigmas,0):min(l + 1 + sigmas,len(data)-1)].sum(axis=0)
    #     newdata.append(aux)
    # data = np.array(newdata)

    for i in range(len(data)):
        if UseMask:
            data[i][Mask] = np.nan
        InitialRMS = np.nanstd(data[i])
        FinalRMS = np.nanstd(data[i][data[i]<5.0*InitialRMS])
        data[i] = data[i]/FinalRMS
    pix1,pix2,pix3 = np.where(data>=MinSN)
    t = Table([pix1, pix3, pix2,data[pix1,pix2,pix3]], names=('Channel', 'Xpix', 'Ypix','SN'))
    t.write(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_pos.fits', format='fits',overwrite=True)
    print 'Positive pixels in search for Sigmas:',sigmas,'N:',len(pix2)


    data = -1.0*data
    pix1,pix2,pix3 = np.where((data)>=MinSN)
    t = Table([pix1, pix3, pix2,data[pix1,pix2,pix3]], names=('Channel', 'Xpix', 'Ypix','SN'))
    t.write(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_neg.fits', format='fits',overwrite=True)
    data = 0
    print 'Negative pixels in search for Sigmas:',sigmas,'N:',len(pix2)


    return

def main():

	#Parse the input arguments
    parser = argparse.ArgumentParser(description="Python script that finds line emission-like features in an ALMA data cube")
    parser.add_argument('-Cube', type=str, required=True,help = 'Path to the Cube fits file where the search will be done')
    parser.add_argument('-OutputPath', type=str, default='OutputLineSearch', required=False , help = 'Directory where the outputs will be saved, if exists the codes finished, otherwise will be created [Default:OutputLineSearch]')
    parser.add_argument('-MinSN', type=float, default = 5.0, required=False,help = 'Minimum S/N value to save in the outputs. A good value depends on each data cube, reasonable values are bettween 3.5 and 6 [Default:5.0]')
    parser.add_argument('-MaxSigmas', type=int, default = 10, required=False,help = 'Maximum number of channels to use as sigma value for the spectral Gaussian convolution. [Default:10]')
    parser.add_argument('-ContinuumImage', type=str, default = 'Continuum.fits', required=False,help = 'Continuum image to use to create a mask. [Default:Continuum.fits]')
    parser.add_argument('-MaskSN', type=float, default = 5.0, required=False,help = 'S/N value to use as limit to create mask from continuum image. [Default:5.0]')
    parser.add_argument('-UseMask', type=str, default = 'False',choices=['True','False'], required=False,help = 'S/N value to use as limit to create mask from continuum image. [Default:5.0]')
    parser.add_argument('-Kernel', type=str, default = 'Gaussian',choices=['Gaussian','gaussian','Tophat','tophat'], required=False,help = 'Type of kernel to use for the convolution. [Default:Gaussian]')
    args = parser.parse_args()

    #Checking input arguments
    print 20*'#','Checking inputs....',20*'#'
    if os.path.exists(args.Cube):
        print '*** Cube',args.Cube,'found ***'
    else:
        print '*** Cube',args.Cube,'not found ***\naborting..'
        exit()


    if os.path.exists(args.OutputPath):
        print '*** Directory',args.OutputPath,'exists ***\naborting..'
        exit()
    else:
        print '*** Creating Directory',args.OutputPath,' ***'
        os.mkdir(args.OutputPath)

 
    if args.MaxSigmas<1:
        print '*** The value for MaxSigmas of',args.MaxSigmas,'is too small ***\naborting..'
        exit()
    else:
        print '*** The value for MaxSigmas of',args.MaxSigmas,'is ok ***'


    if args.MinSN<0:
        print '*** The value for MinSN of',args.MinSN,'has to be positive ***\naborting..'
        exit()
#     elif args.MinSN<3.5:
#     	print '*** The value for MinSN of',args.MinSN,'is smaller than needed but usable, expect larger files ***'
    else:
        print '*** The value for MinSN of',args.MinSN,'is ok ***'

    UseMask = False
    if args.UseMask=='True':
        print '*** Will use Continuum image to create mask ***'
        UseMask = True
    else:
        print '*** Will not use Continuum image to create mask ***'

    if UseMask:
            if os.path.exists(args.ContinuumImage):
                print '*** Continuum Image',args.ContinuumImage,'found ***'
            else:
                print '*** Continuum Image',args.ContinuumImage,'not found ***\naborting..'
                exit()

                if args.MaskSN<0:
                    print '*** The value for MaskSN of',args.MinSN,'has to be positive ***\naborting..'
                    exit()
                else:
                    print '*** The value for MaskSN of',args.MinSN,' is ok ***'

    if args.Kernel=='Gaussian' or args.Kernel=='gaussian':
        print '*** Using Gaussian Kernel ***'
    else:
        print '*** Using Tophat Kernel ***'


    #finding channel width
    Header = fits.open(args.Cube)[0].header
    RefFrequency = Header['CRVAL3']
    ChannelSpacing = Header['CDELT3']
    ApproxChannelVelocityWidth = ( abs(ChannelSpacing)/RefFrequency ) * 3e5
    ApproxMaxSigmas = int ((1000.0/ApproxChannelVelocityWidth) / 2.35) + 1
    if args.Kernel=='Gaussian' or args.Kernel=='gaussian':
        print '*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width FWHM ~ 1000 km/s ***'
    else:
        ApproxMaxSigmas = int ((1000.0/ApproxChannelVelocityWidth))+1
        print '*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width of ~ 1000 km/s for the Tophat Kernel (considering the reference frequency CRVAL3)***'
    try:
        GetValue(args.Cube)
    except:
        print '*** problems reading header ***'
    # Main loop to do the search
    for sigmas in range(args.MaxSigmas):
        SearchLine(args.Cube,args.OutputPath,args.MinSN,sigmas,UseMask,args.ContinuumImage,args.MaskSN,args.Kernel)

main()
