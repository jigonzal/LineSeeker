# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import astropy.io.fits as fits
import os
import scipy.ndimage
import argparse
import os.path

'''

USAGE: "python SearchLine.py -h" will give a description of the input values

python SearchLine_v0.1.py -Cube cube.fits -MaxSigmas 10 -MinSN 3.5 -OutputPath LineSearchTEST1 -UseMask True -ContinuumImage continuum.fits -MaskSN 5.0

Changelog:
---------------------------------------------------------------------------------------------
SearchLine_v0.py
Script that search emission lines features. 

---------------------------------------------------------------------------------------------
SearchLine_v0.1.py
Now it gives the option to use the continuum image to mask continuum sources.

---------------------------------------------------------------------------------------------
SearchLine.py
v0.2
Updated documentation and changed the naming convention where the version will be in the header.


'''

def SearchLine(CubePath,FolderForLinesFiles,MinSN,sigmas,UseMask,ContinuumImage,MaskSN):

    SN = np.array([])
    SNneg = np.array([])

    print 100*'#'
    print 'Starting search of lines with sigma equal to',sigmas,'channels'
    sn_linecandidates_pos = open(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_pos.dat','w')
    sn_linecandidates_neg = open(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_neg.dat','w')


    hdulist =   fits.open(CubePath,memmap=True)
    data  = hdulist[0].data[0]  
    hdulist = 0


    #Nop optimal, but it reads the continuum image every cycle.
    if UseMask:
        DataMask = fits.open(ContinuumImage,memmap=True)[0].data[0][0]  
        InitialRMS = np.nanstd(DataMask)
        FinalRMS = np.nanstd(DataMask[DataMask<MaskSN*InitialRMS])
        Mask = np.where(DataMask>=MaskSN*FinalRMS,True,False)

    data = scipy.ndimage.filters.gaussian_filter(data, [sigmas,0,0],mode='constant', cval=0.0, truncate=4.0)

    for i in range(len(data)):
        if UseMask:
            data[i][Mask] = np.nan
        data[i] = data[i]/np.nanstd(data[i])
    pix1,pix2,pix3 = np.where(data>=MinSN)
    sn_linecandidates_pos.write('-----------------------------------------------------------\n')
    sn_linecandidates_pos.write(' spw0 sigma'+str(sigmas)+'\n')
    sn_linecandidates_pos.write('max_negative_sn: '+str(MinSN)+'\n')
    print 'Positive pixels in search for Sigmas:',sigmas,'N:',len(pix2)
    for k in range(len(pix2)):
      sn_linecandidates_pos.write(str(pix1[k])+' '+str(pix3[k])+' '+str(pix2[k])+' SN:'+str(data[pix1[k],pix2[k],pix3[k]])+'\n')
    sn_linecandidates_pos.close()

    data = -1.0*data
    pix1,pix2,pix3 = np.where((data)>=MinSN)
    sn_linecandidates_neg.write('-----------------------------------------------------------\n')
    sn_linecandidates_neg.write(' spw0 sigma'+str(sigmas)+'\n')
    sn_linecandidates_neg.write('max_negative_sn: '+str(MinSN)+'\n')
    print 'Negative pixels in search for Sigmas:',sigmas,'N:',len(pix2)
    for k in range(len(pix2)):
      sn_linecandidates_neg.write(str(pix1[k])+' '+str(pix3[k])+' '+str(pix2[k])+' SN:'+str(data[pix1[k],pix2[k],pix3[k]])+'\n')
    sn_linecandidates_neg.close()
    data = 0

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
    elif args.MinSN<3.5:
    	print '*** The value for MinSN of',args.MinSN,'is smaller than needed but usable, expect larger files ***'
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


    #finding channel width
    Header = fits.open(args.Cube)[0].header
    RefFrequency = Header['CRVAL3']
    ChannelSpacing = Header['CDELT3']
    ApproxChannelVelocityWidth = ( abs(ChannelSpacing)/RefFrequency ) * 3e5
    ApproxMaxSigmas = int ((1000.0/ApproxChannelVelocityWidth) / 2.35)
    print '*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width FWHM ~ 1000 km/s (considering the reference frequency CRVAL3)***'
    
    # Main loop to do the search
    for sigmas in range(args.MaxSigmas):
        SearchLine(args.Cube,args.OutputPath,args.MinSN,sigmas,UseMask,args.ContinuumImage,args.MaskSN)

main()
