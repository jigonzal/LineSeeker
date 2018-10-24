# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
import os
import scipy.ndimage
import argparse
import os.path
from astropy.convolution import convolve,convolve_fft,Kernel2D
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import LineSeekerFunctions

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

v0.4
Changed the Beam size for the simulations. Now the Simulations use a beam size that has a 68% the 
size of BMAJ and BMIN of the reference beam size in the headers. Now the number of simulationed
detected better resemble the ones detected in the negative and positive data. 
My guess is that this is related to the number of independen beam in an interferometric data (check Condon's paper)

---------------------------------------------------------------------------------------------

v0.5
Change file format for the output, now it uses fits files.
Change size for the reduction of the beam size in simulations. Now it uses sqrt(2) instead of 0.68
New options to change the number of simulated. 

---------------------------------------------------------------------------------------------

v0.6
Changed functions to another file.

'''


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
    parser.add_argument('-Kernel', type=str, default = 'Gaussian',choices=['Gaussian','gaussian','Tophat','tophat'], required=False,help = 'Type of kernel to use for the convolution. [Default:Gaussian]')
    parser.add_argument('-NSimulations', type=int, default = 50, required=False,help = 'Number of cubes to be simulated . [Default:50]')
 
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
#         print '*** The value for MinSN of',args.MinSN,'is smaller than needed but usable, expect larger files ***'
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
    ApproxChannelVelocityWidth = (abs(ChannelSpacing)/RefFrequency)*3e5
    ApproxMaxSigmas = int((1000.0/ApproxChannelVelocityWidth)/2.35) + 1
    if args.Kernel=='Gaussian' or args.Kernel=='gaussian':
        print '*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width FWHM ~ 1000 km/s ***'
    else:
        ApproxMaxSigmas = int ((1000.0/ApproxChannelVelocityWidth))+1
        print '*** MaxSigmas should be of the order of ',ApproxMaxSigmas,'to detect a line width of ~ 1000 km/s for the Tophat Kernel (considering the reference frequency CRVAL3)***'
    

    for SimulationIndex in range(args.NSimulations):
        print '*** Cube: '+str(SimulationIndex+1)+'/'+str(args.NSimulations)+' ***'
        LineSeekerFunctions.SimulateCube(args.Cube)
        os.mkdir(args.OutputPath+'/simul_'+str(SimulationIndex))
        NewPathOutput = args.OutputPath+'/simul_'+str(SimulationIndex)
        # Main loop to do the search
        for sigmas in range(args.MaxSigmas):
            LineSeekerFunctions.SearchLine('SimulatedCube.fits',NewPathOutput,args.MinSN,sigmas,UseMask,args.ContinuumImage,args.MaskSN,args.Kernel)

main()
