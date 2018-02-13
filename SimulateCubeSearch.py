# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import astropy.io.fits as fits
import os
import scipy.ndimage
import argparse
import os.path
from astropy.convolution import convolve,convolve_fft,Kernel2D
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
# import psutil
# pid = os.getpid()
# py = psutil.Process(pid)
# memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
# print 'memory use:', memoryUse
'''

USAGE: "python SimulateCubeSearch.py -h" will give a description of the input values

python SimulateCubeSearch.py -Cube cube.fits -MaxSigmas 10 -MinSN 3.5 -OutputPath Simulation1 -UseMask True -ContinuumImage continuum.fits -MaskSN 5.0


Changelog:
---------------------------------------------------------------------------------------------
SimulateCubeSearch_v0.py
Script that creates simulated cubes given a real one and searches for emisison lines in the noise pure data.
This version works with SearchLine+v0.1.py
python SimulateCubeSearch_v0.py -Cube spw1_w4.fits -UseMask True -ContinuumImage ContinuumLESS1_v2.fits -MaxSigmas 10 -MinSN 3.5 -OutputPath Simulation1 -MaskSN 5.0
---------------------------------------------------------------------------------------------

SimulateCubeSearch_v0.1.py
Script that creates simulated cubes given a real one and searches for emisison lines in the noise pure data.
This version works with SearchLine+v0.1.py
Not the kernel for the beam convolution has a smaller size that is always down to 3 times the sigma of the Gaussian beam.
Now it accepts cubes with one single beam size.
python SimulateCubeSearch_v0.1.py -Cube spw1_w4.fits -UseMask True -ContinuumImage ContinuumLESS1_v2.fits -MaxSigmas 10 -MinSN 3.5 -OutputPath Simulation1 -MaskSN 5.0
---------------------------------------------------------------------------------------------

SimulateCubeSearch_v0.2.py
small changes made to try to improve memory usage.
python SimulateCubeSearch_v0.2.py -Cube spw1_w4.fits -UseMask True -ContinuumImage ContinuumLESS1_v2.fits -MaxSigmas 10 -MinSN 3.5 -OutputPath Simulation1 -MaskSN 5.0
---------------------------------------------------------------------------------------------

v0.3
Updated documentation and changed the naming convention where the version will be in the header.
Remove printing of memory usage
---------------------------------------------------------------------------------------------

'''

def SimulateCube(CubePath):
#     memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
#     print 'memory use:', memoryUse
    print 100*'#'
    print 'Creating Simulated Cube...'
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
    factor = 1.0/factor
    # hdulist.close()

    KernelList = []
    for i in range(len(BMAJ)):
        SigmaPixel = int((BMAJ[i]/2.355)/pix_size)

        x = np.arange(-(3*SigmaPixel), (3*SigmaPixel)+1)
        y = np.arange(-(3*SigmaPixel), (3*SigmaPixel)+1)
        x, y = np.meshgrid(x, y)
        arr = models.Gaussian2D(amplitude=1.0,x_mean=0,y_mean=0,x_stddev=(BMAJ[i]/2.355)/pix_size,y_stddev=(BMIN[i]/2.355)/pix_size,theta=(BPA[i]*2.0*np.pi/360.0)+np.pi/2)(x,y)
        kernel = Kernel2D(model=models.Gaussian2D(amplitude=1.0,x_mean=0,y_mean=0,x_stddev=(BMAJ[i]/2.355)/pix_size,y_stddev=(BMIN[i]/2.355)/pix_size,theta=(BPA[i]*2.0*np.pi/360.0)+np.pi/2),array=arr,width=len(x))
        KernelList.append(kernel)

    RMS = []
    for i in range(len(data)):
        InitialRMS = np.nanstd(data[i])
        FinalRMS = np.nanstd(data[i][data[i]<5.0*InitialRMS])
        RMS.append(FinalRMS)
    RMS = np.array(RMS)
    print 'Average RMS per channel:',np.mean(RMS[np.isfinite(RMS)]),'Jy/beam'
    print 'Median RMS per channel:',np.median(RMS[np.isfinite(RMS)]),'Jy/beam'


    RandomNoiseCube = np.random.normal(size=np.shape(data)) 

    for i in range(len(RandomNoiseCube)):
        smoothed = convolve_fft(RandomNoiseCube[i], KernelList[i])
        std_aux = np.nanstd(smoothed)
        if np.isfinite(RMS[i]) and RMS[i]!=0.0:
            RandomNoiseCube[i] = smoothed*RMS[i]/std_aux
        else:
            RandomNoiseCube[i] = 0.0
        RandomNoiseCube[i][np.isnan(data[i])] = np.nan



    hdulist[0].data[0] = RandomNoiseCube
    hdulist.writeto('SimulatedCube.fits',overwrite=True)
    hdulist.close()
    hdulist = None
    data = None
#     memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
#     print 'memory use:', memoryUse	
    RandomNoiseCube = None
    # return RandomNoiseCube
    return RandomNoiseCube


def SearchLine(FolderForLinesFiles,MinSN,sigmas,UseMask,ContinuumImage,MaskSN):
#     memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
#     print 'memory use:', memoryUse	
    SN = np.array([])
    SNneg = np.array([])

    print 100*'#'
    print 'Starting search of lines with sigma equal to',sigmas,'channels'
    sn_linecandidates_pos = open(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_pos.dat','w')
    sn_linecandidates_neg = open(FolderForLinesFiles+'/line_dandidates_sn_sigmas'+str(sigmas)+'_neg.dat','w')


    hdulist =   fits.open('SimulatedCube.fits',memmap=True)
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
        InitialRMS = np.nanstd(data[i])
        FinalRMS = np.nanstd(data[i][data[i]<5.0*InitialRMS])
        data[i] = data[i]/FinalRMS
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
#     memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
#     print 'memory use:', memoryUse	
    return

def main():

    #Parse the input arguments
    parser = argparse.ArgumentParser(description="Python script that finds line emission-like features in an ALMA data cube")
    parser.add_argument('-Cube', type=str, required=True,help = 'Path to the Cube fits file where the search will be done')
    parser.add_argument('-OutputPath', type=str, default='Simulation', required=False , help = 'Directory where the outputs will be saved, if exists the codes finished, otherwise will be created [Default:Simulation]')
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
    ApproxMaxSigmas = int ((1000.0/ApproxChannelVelocityWidth) / 2.35) + 1
    print '*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width FWHM ~ 1000 km/s ***'
    

    NumberOfSimulations = 50

    for SimulationIndex in range(NumberOfSimulations):
        # SimulatedCube = SimulateCube(args.Cube)
        SimulateCube(args.Cube)
        os.mkdir(args.OutputPath+'/simul_'+str(SimulationIndex))
        NewPathOutput = args.OutputPath+'/simul_'+str(SimulationIndex)
        # Main loop to do the search
        for sigmas in range(args.MaxSigmas):
            SearchLine(NewPathOutput,args.MinSN,sigmas,UseMask,args.ContinuumImage,args.MaskSN)

main()
