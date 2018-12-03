import os
import numpy as np
import astropy.io.fits as fits
import scipy.ndimage
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy.convolution import convolve,convolve_fft,Kernel2D
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit

'''
Function used by LineSeeker. 
'''

def GetMinSNEstimate(CubePath):
    '''
    Function that tried to get a rough estimate to the MinSN to use for the search. 
    This is very difficul to only use it as a rough reference. It is known to fail and give
    totally worng estimate in many cases.
    '''
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
    factor = 2.0*(np.pi*BMAJ*BMIN/(8.0*np.log(2)))/(pix_size**2)
    RefFrequency = head['CRVAL3']
    ChannelSpacing = head['CDELT3']
    ApproxChannelVelocityWidth = (abs(ChannelSpacing)/RefFrequency)*3e5
    ApproxMaxSigmas = 1000.0/ApproxChannelVelocityWidth
    aux = len(data[0][np.isfinite(data[0])].flatten())*1.0/factor[0]*(len(data)/ApproxMaxSigmas)
    Number2Print = round(np.power(10,np.log10(aux)*0.07723905 + 0.19291493),1)
    print '*** A rough guesstimate to use as MinSN is',Number2Print,'***'
    # print len(data[0][np.isfinite(data[0])].flatten())*1.0/factor[0]*(len(data)/ApproxMaxSigmas)
    return 

def SearchLine(CubePath,FolderForLinesFiles,MinSN,sigmas,UseMask,ContinuumImage,MaskSN,Kernel):

    '''
    Function that search for emission lines.
    '''
    SN = np.array([])
    SNneg = np.array([])

    print 100*'#'
    print 'Starting search of lines with parameter for filter equal to',sigmas,'channels'

    hdulist = fits.open(CubePath,memmap=True)
    data  = hdulist[0].data[0]  
    hdulist = 0

    #Nop optimal, but it reads the continuum image every cycle.
    if UseMask:
        DataMask = fits.open(ContinuumImage,memmap=True)[0].data[0][0]  
        InitialRMS = np.nanstd(DataMask)
        FinalRMS = np.nanstd(DataMask[DataMask<MaskSN*InitialRMS])
        Mask = np.where(DataMask>=MaskSN*FinalRMS,True,False)


    if Kernel=='Gaussian' or Kernel=='guassian':
        data = scipy.ndimage.filters.gaussian_filter(data, [sigmas,0,0],mode='constant',
        																cval=0.0, 
        																truncate=4.0)
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

def SimulateCube(CubePath):
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
    FractionBeam = 1.0/np.sqrt(2.0)
    print 'Fraction Beam',FractionBeam
    KernelList = []

    for i in range(len(BMAJ)):
        SigmaPixel = int((BMAJ[i]*FractionBeam/2.355)/pix_size)+1
        x = np.arange(-(3*SigmaPixel), (3*SigmaPixel))
        y = np.arange(-(3*SigmaPixel), (3*SigmaPixel))        
        x, y = np.meshgrid(x, y)
        arr = models.Gaussian2D(amplitude=1.0,x_mean=0,y_mean=0,
        						x_stddev=(BMAJ[i]*FractionBeam/2.355)/pix_size,
        						y_stddev=(BMIN[i]*FractionBeam/2.355)/pix_size,
        						theta=(BPA[i]*2.0*np.pi/360.0)+np.pi/2)(x,y)
        kernel = Kernel2D(model=models.Gaussian2D(amplitude=1.0,x_mean=0,y_mean=0,
        					x_stddev=(BMAJ[i]*FractionBeam/2.355)/pix_size,
        					y_stddev=(BMIN[i]*FractionBeam/2.355)/pix_size,
        					theta=(BPA[i]*2.0*np.pi/360.0)+np.pi/2),
        					array=arr,width=len(x))
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
        if np.isfinite(RMS[i]) and RMS[i]!=0.0:
            smoothed = convolve_fft(RandomNoiseCube[i], KernelList[i],allow_huge=True)
            std_aux = np.nanstd(smoothed)
            RandomNoiseCube[i] = smoothed*RMS[i]/std_aux
        else:
            RandomNoiseCube[i] = 0.0
        RandomNoiseCube[i][np.isnan(data[i])] = np.nan

    hdulist[0].data[0] = RandomNoiseCube
    hdulist.writeto('SimulatedCube.fits',overwrite=True)
    hdulist.close()
    hdulist = None
    data = None
    RandomNoiseCube = None
    return RandomNoiseCube

def GetPixelsPerBMAJ(CubePath):
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
    return max(BMAJ/pix_size)

def NegativeRate(SNR,N,sigma):
	# return N*np.exp(-1.0*np.power(SNR,2)/(2.0*np.power(sigma,2)))
	return N*0.5 *( 1.0 -  scipy.special.erf(SNR/(np.sqrt(2.0)*sigma)))  #1 - CDF(SNR) assuming Gaussian distribution and N independent elements.

def NegativeRateLog(SNR,N,sigma):
	# return N*np.exp(-1.0*np.power(SNR,2)/(2.0*np.power(sigma,2)))
	return np.log10(N*0.5 *( 1.0 -  scipy.special.erf(SNR/(np.sqrt(2.0)*sigma))))  #1 - CDF(SNR) assuming Gaussian distribution and N independent elements.

def GetFinalSN(SourcesTotal,PixelsPerBMAJ):
	COORD = []
	X = []
	Y = []
	Channel = []
	SN_array = []
	purity = []
	for NewSource in SourcesTotal:
			COORD.append(np.array([NewSource[1],NewSource[2],NewSource[0]]))
			X.append(NewSource[1])
			Y.append(NewSource[2])
			Channel.append(NewSource[0])
			SN_array.append(NewSource[3])		
			purity.append(NewSource[5])	

	COORD = np.array(COORD)
	X = np.array(X)
	Y = np.array(Y)
	Channel = np.array(Channel)
	SN = np.array(SN_array)
	purity = np.array(purity)

	db = DBSCAN(eps=PixelsPerBMAJ, min_samples=1,leaf_size=30).fit(COORD)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	unique_labels = set(labels)


	FinalSN = []
	for k in unique_labels:
		class_member_mask = (labels == k)
		FinalSN.append(max(SN[class_member_mask]))


	FinalSN = np.array(FinalSN)
	FinalSN = FinalSN[np.argsort(FinalSN)]
	return FinalSN

def GetArraysFromFile(path,Sources,minSN):
	path = path.replace('.fits','').replace('.dat','')

	try:
		table = fits.open(path+'.fits')[1].data
		SN = table['SN']
		X = table['Xpix'][SN>=minSN]
		Y = table['Ypix'][SN>=minSN]
		Channel = table['Channel'][SN>=minSN]
		SN_array = table['SN'][SN>=minSN]
		COORD = np.transpose(np.array([X,Y,Channel]))
		COORD = list(COORD)

		for source in Sources:
				COORD.append(np.array([source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[0][np.argmax(source[3])]]))
				X = np.append(X,source[1][np.argmax(source[3])])
				Y = np.append(Y,source[2][np.argmax(source[3])])
				Channel = np.append(Channel,source[0][np.argmax(source[3])])
				SN_array = np.append(SN_array,source[3][np.argmax(source[3])])
		
		COORD = np.array(COORD)
		X = np.array(X)
		Y = np.array(Y)
		Channel = np.array(Channel)
		SN = np.array(SN_array)
		return COORD,X,Y,Channel,SN
	except:
		COORD = []
		X = []
		Y = []
		Channel = []
		SN_array = []
		FileReader = open(path+'.dat').readlines()

		for j in FileReader:
			FirstCharacter = j[0]
			j = j.split()
			if FirstCharacter == ' ':
				continue
			if FirstCharacter == '-' or j[0]== 'max_negative_sn:':
				continue
			SN = np.float(j[3].replace('SN:',''))

			if SN>=minSN :
				spw = int(j[0])
				x = np.float(j[1])
				y = np.float(j[2])
				sn = np.float(j[-1].replace('SN:',''))
				COORD.append(np.array([x,y,spw]))
				X.append(x)
				Y.append(y)
				Channel.append(spw)
				SN_array.append(sn)

		for source in Sources:
				COORD.append(np.array([source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[0][np.argmax(source[3])]]))
				X = np.append(X,source[1][np.argmax(source[3])])
				Y = np.append(Y,source[2][np.argmax(source[3])])
				Channel = np.append(Channel,source[0][np.argmax(source[3])])
				SN_array = np.append(SN_array,source[3][np.argmax(source[3])])

		COORD = np.array(COORD)
		X = np.array(X)
		Y = np.array(Y)
		Channel = np.array(Channel)
		SN = np.array(SN_array)
		return COORD,X,Y,Channel,SN

def GetSourcesFromFiles(files,minSN,PixelsPerBMAJ):
	files.sort()
	Sources = []
	SourcesAux = []

	for i in files:
		COORD,X,Y,Channel,SN = GetArraysFromFile(i,Sources,minSN)
		if len(COORD)>0:
			db = DBSCAN(eps=PixelsPerBMAJ, min_samples=1,leaf_size=30).fit(COORD)
			core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
			core_samples_mask[db.core_sample_indices_] = True
			labels = db.labels_
			unique_labels = set(labels)
			SourcesAux = []

			for k in unique_labels:
			    class_member_mask = (labels == k)
			    source = [Channel[class_member_mask],X[class_member_mask],Y[class_member_mask],SN[class_member_mask],max(Channel[class_member_mask])-min(Channel[class_member_mask])]
			    SourcesAux.append(source)
		else:
			SourcesAux = []

		for source in SourcesAux:
			Sources.append(source)
	return Sources

def GetFinalCandidates(SourcesTotalPos,PixelsPerBMAJ):
	COORD = []
	X = []
	Y = []
	Channel = []
	SN_array = []
	puritySimulation= []
	purityNegative = []
	purityPoisson = []
	psimulationE1 = []
	psimulationE2 = []
	ppoissonE1 = []
	ppoissonE2 = []	
	purityNegativeE1 = []
	purityNegativeE2 = []
	pNegDiv = []
	pNegDivE1 = []
	pNegDivE2 = []
	pSimExp = []
	pSimExpE1 = []
	pSimExpE2 = []
	pPoiExp = []
	pPoiExpE1 = []
	pPoiExpE2 = []

	for NewSource in SourcesTotalPos:
			COORD.append(np.array([NewSource[1],NewSource[2],NewSource[0]]))
			X.append(NewSource[1])
			Y.append(NewSource[2])
			Channel.append(NewSource[0])
			SN_array.append(NewSource[3])		
			puritySimulation.append(NewSource[5])	
			purityNegative.append(NewSource[6])	
			purityPoisson.append(NewSource[7])	
			psimulationE1.append(NewSource[8])	
			psimulationE2.append(NewSource[9])	
			ppoissonE1.append(NewSource[10])	
			ppoissonE2.append(NewSource[11])	
			purityNegativeE1.append(NewSource[12])
			purityNegativeE2.append(NewSource[13])
			pNegDiv.append(NewSource[14])
			pNegDivE1.append(NewSource[15])
			pNegDivE2.append(NewSource[16])
			pSimExp.append(NewSource[17])
			pSimExpE1.append(NewSource[18])
			pSimExpE2.append(NewSource[19])
			pPoiExp.append(NewSource[20])
			pPoiExpE1.append(NewSource[21])
			pPoiExpE2.append(NewSource[22])

	COORD = np.array(COORD)
	X = np.array(X)
	Y = np.array(Y)
	Channel = np.array(Channel)
	SN = np.array(SN_array)
	puritySimulation = np.array(puritySimulation)
	purityNegative = np.array(purityNegative)
	purityPoisson = np.array(purityPoisson)
	psimulationE1 = np.array(psimulationE1)
	psimulationE2 = np.array(psimulationE2)
	ppoissonE1 = np.array(ppoissonE1)
	ppoissonE2 = np.array(ppoissonE2)
	purityNegativeE1 = np.array(purityNegativeE1)
	purityNegativeE2 = np.array(purityNegativeE2)
	pNegDiv = np.array(pNegDiv)
	pNegDivE1 = np.array(pNegDivE1)
	pNegDivE2 = np.array(pNegDivE2)
	pSimExp = np.array(pSimExp)
	pSimExpE1 = np.array(pSimExpE1)
	pSimExpE2 = np.array(pSimExpE2)
	pPoiExp = np.array(pPoiExp)
	pPoiExpE1 = np.array(pPoiExpE1)
	pPoiExpE2 = np.array(pPoiExpE2)

	db = DBSCAN(eps=PixelsPerBMAJ, min_samples=1,leaf_size=30).fit(COORD)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	unique_labels = set(labels)

	FinalX = []
	FinalY = []
	FinalChannel = []
	FinalSN = []
	FinalPuritySimulation = []
	FinalPurityNegative = []
	FinalPurityPoisson = []
	FinalPSimultionE1 = []
	FinalPSimultionE2 = []	
	FinalPPoissonE1 = []
	FinalPPoissonE2 = []
	FinalPuritySimulationE1 = []
	FinalPuritySimulationE2 = []
	FinalpNegDiv = []
	FinalpNegDivE1 = []
	FinalpNegDivE2 = []
	FinalpSimExp = []
	FinalpSimExpE1 = []
	FinalpSimExpE2 = []
	FinalpPoiExp = []
	FinalpPoiExpE1 = []
	FinalpPoiExpE2 = []

	for k in unique_labels:
		class_member_mask = (labels == k)
		FinalX.append(X[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalY.append(Y[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalChannel.append(Channel[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalSN.append(max(SN[class_member_mask]))
		FinalPuritySimulation.append(min(puritySimulation[class_member_mask]))
		FinalPurityNegative.append(min(min(purityNegative[class_member_mask]),1))
		FinalPurityPoisson.append(min(purityPoisson[class_member_mask]))
		FinalPSimultionE1.append(psimulationE1[class_member_mask][np.argmin(puritySimulation[class_member_mask])])
		FinalPSimultionE2.append(psimulationE2[class_member_mask][np.argmin(puritySimulation[class_member_mask])])
		FinalPPoissonE1.append(ppoissonE1[class_member_mask][np.argmin(purityPoisson[class_member_mask])])
		FinalPPoissonE2.append(ppoissonE2[class_member_mask][np.argmin(purityPoisson[class_member_mask])])
		FinalPuritySimulationE1.append(purityNegativeE1[class_member_mask][np.argmin(purityNegative[class_member_mask])])
		FinalPuritySimulationE2.append(purityNegativeE2[class_member_mask][np.argmin(purityNegative[class_member_mask])])
		FinalpNegDiv.append(min(pNegDiv[class_member_mask]))
		FinalpNegDivE1.append(pNegDivE1[class_member_mask][np.argmin(pNegDiv[class_member_mask])])
		FinalpNegDivE2.append(pNegDivE2[class_member_mask][np.argmin(pNegDiv[class_member_mask])])
		FinalpSimExp.append(min(pSimExp[class_member_mask]))
		FinalpSimExpE1.append(pSimExpE1[class_member_mask][np.argmin(pSimExp[class_member_mask])])
		FinalpSimExpE2.append(pSimExpE2[class_member_mask][np.argmin(pSimExp[class_member_mask])])
		FinalpPoiExp.append(min(pPoiExp[class_member_mask]))
		FinalpPoiExpE1.append(pPoiExpE1[class_member_mask][np.argmin(pPoiExp[class_member_mask])])
		FinalpPoiExpE2.append(pPoiExpE2[class_member_mask][np.argmin(pPoiExp[class_member_mask])])	


	FinalX = np.array(FinalX)
	FinalY = np.array(FinalY)
	FinalChannel = np.array(FinalChannel)
	FinalSN = np.array(FinalSN)
	FinalPuritySimulation = np.array(FinalPuritySimulation)
	FinalPurityNegative = np.array(FinalPurityNegative)
	FinalPurityPoisson = np.array(FinalPurityPoisson)
	FinalPSimultionE1 = np.array(FinalPSimultionE1)
	FinalPSimultionE2 = np.array(FinalPSimultionE2)
	FinalPPoissonE1 = np.array(FinalPPoissonE1)
	FinalPPoissonE2 = np.array(FinalPPoissonE2)
	FinalPuritySimulationE1 = np.array(FinalPuritySimulationE1)
	FinalPuritySimulationE2 = np.array(FinalPuritySimulationE2)
	FinalpNegDiv = np.array(FinalpNegDiv)
	FinalpNegDivE1 = np.array(FinalpNegDivE1)
	FinalpNegDivE2 = np.array(FinalpNegDivE2)
	FinalpSimExp = np.array(FinalpSimExp)
	FinalpSimExpE1 = np.array(FinalpSimExpE1)
	FinalpSimExpE2 = np.array(FinalpSimExpE2)
	FinalpPoiExp = np.array(FinalpPoiExp)
	FinalpPoiExpE1 = np.array(FinalpPoiExpE1)
	FinalpPoiExpE2 = np.array(FinalpPoiExpE2)


	FinalX = FinalX[np.argsort(FinalSN)]
	FinalY = FinalY[np.argsort(FinalSN)]
	FinalChannel = FinalChannel[np.argsort(FinalSN)]
	FinalPuritySimulation = FinalPuritySimulation[np.argsort(FinalSN)]
	FinalPurityNegative = FinalPurityNegative[np.argsort(FinalSN)]
	FinalPurityPoisson = FinalPurityPoisson[np.argsort(FinalSN)]
	FinalPSimultionE1 = FinalPSimultionE1[np.argsort(FinalSN)]
	FinalPSimultionE2 = FinalPSimultionE2[np.argsort(FinalSN)]
	FinalPPoissonE1 = FinalPPoissonE1[np.argsort(FinalSN)]
	FinalPPoissonE2 = FinalPPoissonE2[np.argsort(FinalSN)]
	FinalPuritySimulationE1 = np.array(FinalPuritySimulationE1)[np.argsort(FinalSN)]
	FinalPuritySimulationE2 = np.array(FinalPuritySimulationE2)[np.argsort(FinalSN)]
	FinalpNegDiv = np.array(FinalpNegDiv)[np.argsort(FinalSN)]
	FinalpNegDivE1 = np.array(FinalpNegDivE1)[np.argsort(FinalSN)]
	FinalpNegDivE2 = np.array(FinalpNegDivE2)[np.argsort(FinalSN)]
	FinalpSimExp = np.array(FinalpSimExp)[np.argsort(FinalSN)]
	FinalpSimExpE1 = np.array(FinalpSimExpE1)[np.argsort(FinalSN)]
	FinalpSimExpE2 = np.array(FinalpSimExpE2)[np.argsort(FinalSN)]
	FinalpPoiExp = np.array(FinalpPoiExp)[np.argsort(FinalSN)]
	FinalpPoiExpE1 = np.array(FinalpPoiExpE1)[np.argsort(FinalSN)]
	FinalpPoiExpE2 = np.array(FinalpPoiExpE2)[np.argsort(FinalSN)]
	FinalSN = FinalSN[np.argsort(FinalSN)]

	output = [
				FinalX,
				FinalY,
				FinalChannel,
				FinalPuritySimulation,
				FinalPurityNegative,
				FinalPurityPoisson,
				FinalSN,
				FinalPSimultionE1,
				FinalPSimultionE2,
				FinalPPoissonE1,
				FinalPPoissonE2,
				FinalPuritySimulationE1,
				FinalPuritySimulationE2,
				FinalpNegDiv,
				FinalpNegDivE1,
				FinalpNegDivE2,
				FinalpSimExp,
				FinalpSimExpE1,
				FinalpSimExpE2,
				FinalpPoiExp,
				FinalpPoiExpE1,
				FinalpPoiExpE2
			]
	return output

def GetPoissonErrorGivenMeasurements(NMeasured,Total):
	k = NMeasured
	n = Total
	aux = scipy.special.betaincinv(k+1.0, n+1.0-k, [0.16,0.5,0.84])
	Estimate = 1.0*NMeasured/Total
	E1 = aux[1]-aux[0]
	E2 = aux[2]-aux[1]
	return Estimate,E1,E2

def GetPoissonEstimates(bins,SNFinalPos,SNFinalNeg,LimitN,MinSN):

	ProbPoisson = []
	ProbPoissonE1 = []
	ProbPoissonE2 = []
	ProbNegativeOverPositive = []
	ProbNegativeOverPositiveE1 = []
	ProbNegativeOverPositiveE2 = []
	ProbPoissonExpected = []
	ProbPoissonExpectedE1 = []
	ProbPoissonExpectedE2 = []
	ProbNegativeOverPositiveDif = []
	ProbNegativeOverPositiveDifE1 = []
	ProbNegativeOverPositiveDifE2 = []	
	PurityPoisson = []
	Nnegative = []
	NnegativeReal = []
	NPositive = []
	Nnegative_e1 = []
	Nnegative_e2 = []

	for sn in bins:
		if len(SNFinalPos[SNFinalPos>=sn])>0:
			Fraction,FractionE1,FractionE2 = GetPoissonErrorGivenMeasurements(len(SNFinalNeg[SNFinalNeg>=sn]),len(SNFinalPos[SNFinalPos>=sn]))

			if Fraction>1.0:
				Fraction = 1.0
				FractionE1 = 0.0
				FractionE2 = 0.0
			else:
				pass

			ProbNegativeOverPositive.append(Fraction)
			ProbNegativeOverPositiveE1.append(FractionE1)
			ProbNegativeOverPositiveE2.append(FractionE2)
		elif len(SNFinalNeg[SNFinalNeg>=sn])>0:
			ProbNegativeOverPositive.append(1.0)
			ProbNegativeOverPositiveE1.append(0.0)
			ProbNegativeOverPositiveE2.append(0.0)
		else:
			ProbNegativeOverPositive.append(0.0)
			ProbNegativeOverPositiveE1.append(0.0)
			ProbNegativeOverPositiveE2.append(0.0)

		if len(SNFinalPos[(SNFinalPos>=sn) & (SNFinalPos<sn+0.1)])>0:
			Fraction,FractionE1,FractionE2 = GetPoissonErrorGivenMeasurements(len(SNFinalNeg[(SNFinalNeg>=sn) & (SNFinalNeg<sn+0.1)]),len(SNFinalPos[(SNFinalPos>=sn) & (SNFinalPos<sn+0.1)]))
			if Fraction>1.0:
				Fraction = 1.0
				FractionE1 = 0.0
				FractionE2 = 0.0
			else:
				pass

			ProbNegativeOverPositiveDif.append(min(1.0,Fraction))
			ProbNegativeOverPositiveDifE1.append(FractionE1)
			ProbNegativeOverPositiveDifE2.append(FractionE2)
		elif len(SNFinalNeg[(SNFinalNeg>=sn) & (SNFinalNeg<sn+0.1)])>0:
			ProbNegativeOverPositiveDif.append(1.0)
			ProbNegativeOverPositiveDifE1.append(0.0)
			ProbNegativeOverPositiveDifE2.append(0.0)
		else:
			ProbNegativeOverPositiveDif.append(0.0)
			ProbNegativeOverPositiveDifE1.append(0.0)
			ProbNegativeOverPositiveDifE2.append(0.0)

		k = len(SNFinalNeg[SNFinalNeg>=sn])
		aux = scipy.special.gammaincinv(k + 1, [0.16,0.5,0.84])
		NnegativeReal.append(k)
		Nnegative.append(aux[1])
		Nnegative_e1.append(aux[1]-aux[0])
		Nnegative_e2.append(aux[2]-aux[1])
		NPositive.append(1.0*len(SNFinalPos[SNFinalPos>=sn]))

	Nnegative = np.array(Nnegative)
	NPositive = np.array(NPositive)
	NnegativeReal = np.array(NnegativeReal)
	Nnegative_e1 = np.array(Nnegative_e1)
	Nnegative_e2 = np.array(Nnegative_e2)
	

	MinSNtoFit = min(bins)
	UsableBins = len(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN])

	print 'Min SN to do the fit:',round(MinSNtoFit,1),', Number of usable bins:',UsableBins
	if UsableBins<6:
		print '*** We are using ',UsableBins,' points for the fitting of the negative counts ***'
		print '*** We usually get good results with 6 points, try reducing the parameter -MinSN ***'
	while UsableBins>6:
		MinSNtoFit = MinSNtoFit + 0.1
		UsableBins = len(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN])
		print 'Min SN to do the fit:',round(MinSNtoFit,1),', Number of usable bins:',UsableBins
		if MinSNtoFit>max(bins):
			print 'No negative points to do the fit'
			exit()


	if UsableBins>=3:
		try:
			# popt, pcov = curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],p0=[1e6,1])
			popt, pcov = curve_fit(NegativeRateLog, 
					bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],
					np.log10(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]),
					p0=[1e6,1],
					sigma=np.log10(np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0)),
					absolute_sigma=False)

			perr = np.sqrt(np.diag(pcov))
			# print popt,popt/perr,not np.isfinite(perr[0])
			CounterFitTries = 0
			while not np.isfinite(perr[0]):
				print '*** curve_fit failed to converge ... ***'
				NewParameter1 = np.power(10,np.random.uniform(1,9))
				NewParameter2 = np.random.uniform(0.1,2.0)
				print '*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***'
				# popt, pcov = curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],p0=[NewParameter1,NewParameter2])
				popt, pcov = curve_fit(NegativeRateLog, 
										bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],
										np.log10(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]),
										p0=[NewParameter1,NewParameter2],
										sigma=np.log10(np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0)),
										absolute_sigma=False)
				perr = np.sqrt(np.diag(pcov))
				print '*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***'
				CounterFitTries += 1
				if CounterFitTries >100:
					print '*** Over 100 attemps and no good fit *** '
					break

		except:
			print 'Fitting failed for LimitN:'+str(LimitN)+' and '+str(MinSN)+'... Will force LimitN=0'
			# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[1e6,1])	
			popt, pcov = curve_fit(NegativeRateLog, 
					bins[Nnegative>0],
					np.log10(Nnegative[Nnegative>0]),
					p0=[1e6,1],
					sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
					absolute_sigma=False)
			perr = np.sqrt(np.diag(pcov))
			# print popt,popt/perr,not np.isfinite(perr[0])
			CounterFitTries = 0
			while not np.isfinite(perr[0]):
				print '*** curve_fit failed to converge ... ***'
				NewParameter1 = np.power(10,np.random.uniform(1,9))
				NewParameter2 = np.random.uniform(0.1,2.0)
				print '*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***'
				# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[NewParameter1,NewParameter2])
				popt, pcov = curve_fit(NegativeRateLog, 
										bins[Nnegative>0],
										np.log10(Nnegative[Nnegative>0]),
										p0=[NewParameter1,NewParameter2],
										sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
										absolute_sigma=False)
				perr = np.sqrt(np.diag(pcov))
				print '*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***'
				CounterFitTries += 1
				if CounterFitTries >100:
					print '*** Over 100 attemps and no good fit *** '
					break
	else:
		print 'Number of usable bins is less than 3 for LimitN:'+str(LimitN)+' and '+str(MinSN)+'... Will force LimitN=0'
		# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[1e6,1])	
		popt, pcov = curve_fit(NegativeRateLog, 
					bins[Nnegative>0],
					np.log10(Nnegative[Nnegative>0]),
					p0=[1e6,1],
					sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
					absolute_sigma=False)
		perr = np.sqrt(np.diag(pcov))
		# print popt,popt/perr,not np.isfinite(perr[0])
		CounterFitTries = 0
		while not np.isfinite(perr[0]):
			print '*** curve_fit failed to converge ... ***'
			NewParameter1 = np.power(10,np.random.uniform(1,9))
			NewParameter2 = np.random.uniform(0.1,2.0)
			print '*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***'
			# popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[NewParameter1,NewParameter2])
			popt, pcov = curve_fit(NegativeRateLog, 
										bins[Nnegative>0],
										np.log10(Nnegative[Nnegative>0]),
										p0=[NewParameter1,NewParameter2],
										sigma=np.log10(np.average([Nnegative_e1[Nnegative>0],Nnegative_e2[Nnegative>0]],axis=0)),
										absolute_sigma=False)
			perr = np.sqrt(np.diag(pcov))
			print '*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***'
			CounterFitTries += 1
			if CounterFitTries >100:
				print '*** Over 100 attemps and no good fit *** '
				break

	NegativeFitted = NegativeRate(bins,popt[0],popt[1])
	SNPeakGaussian = (popt/np.sqrt(np.diag(pcov)))[0]
	# print 'SNPeakGaussian',SNPeakGaussian,popt,np.sqrt(np.diag(pcov))
	# print curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],p0=[1e6,1],sigma=np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0),absolute_sigma=False)
	# print curve_fit(NegativeRateLog, 
	# 				bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],
	# 				np.log10(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]),
	# 				p0=[1e6,1],
	# 				sigma=np.log10(np.average([Nnegative_e1[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN],Nnegative_e2[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>LimitN]],axis=0)),
	# 				absolute_sigma=False)

	for i in range(len(bins)):
		aux = []
		auxExpected = []
		for j in range(1000):
			lamb = np.random.normal(NegativeFitted[i],NegativeFitted[i]/SNPeakGaussian)
			while lamb<0:
				lamb = np.random.normal(NegativeFitted[i],NegativeFitted[i]/SNPeakGaussian)
			aux.append(1-scipy.special.gammaincc(0+1,lamb))
			if i ==len(bins)-1:
				if NPositive[i]>0:
					auxExpected.append(1.0-max(0,NPositive[i]-lamb)/NPositive[i])
				else:
					auxExpected.append(0.0)
			else:
				# lamb2 = lamb - np.random.normal(NegativeFitted[i+1],NegativeFitted[i+1]/SNPeakGaussian) 
				lamb2 = (NegativeFitted[i] - NegativeFitted[i+1])*lamb/NegativeFitted[i]
				while lamb2<0:
					lamb2 = lamb - np.random.normal(NegativeFitted[i+1],NegativeFitted[i+1]/SNPeakGaussian) 
				if (NPositive[i] - NPositive[i+1])>0:
					auxExpected.append(1.0-max(0,(NPositive[i] - NPositive[i+1]) - lamb2)/(NPositive[i] - NPositive[i+1]))
				else:
					auxExpected.append(0.0)
					# auxExpected.append(1.0-max(0,0.7 - lamb2)/0.7)


		PP = np.nanpercentile(aux,[16,50,84])
		PPExpected = np.nanpercentile(auxExpected,[16,50,84])
		ProbPoisson.append(PP[1])
		ProbPoissonE1.append(PP[1]-PP[0])
		ProbPoissonE2.append(PP[2]-PP[1])

		ProbPoissonExpected.append(PPExpected[1])
		ProbPoissonExpectedE1.append(PPExpected[1]-PPExpected[0])
		ProbPoissonExpectedE2.append(PPExpected[2]-PPExpected[1])		
		# if i<len(bins)-1:
		# 	print bins[i],PPExpected,NegativeFitted[i],NPositive[i],NPositive[i+1]
		if NPositive[i]>0:
			PurityPoisson.append(max((NPositive[i]-NegativeFitted[i])/NPositive[i],0))
		else:
			PurityPoisson.append(0.0)

	ProbPoisson = np.array(ProbPoisson)
	ProbPoissonE1 = np.array(ProbPoissonE1)
	ProbPoissonE2 = np.array(ProbPoissonE2)
	ProbNegativeOverPositive = np.array(ProbNegativeOverPositive)
	ProbNegativeOverPositiveE1 = np.array(ProbNegativeOverPositiveE1)
	ProbNegativeOverPositiveE2 = np.array(ProbNegativeOverPositiveE2)
	ProbNegativeOverPositiveDif = np.array(ProbNegativeOverPositiveDif)
	ProbNegativeOverPositiveDifE1 = np.array(ProbNegativeOverPositiveDifE1)
	ProbNegativeOverPositiveDifE2 = np.array(ProbNegativeOverPositiveDifE2)
	ProbPoissonExpected = np.array(ProbPoissonExpected)
	ProbPoissonExpectedE1 = np.array(ProbPoissonExpectedE1)
	ProbPoissonExpectedE2 = np.array(ProbPoissonExpectedE2)
	PurityPoisson = np.array(PurityPoisson)

	output = [
				bins,
				ProbPoisson,
				ProbNegativeOverPositive,
				PurityPoisson,
				NPositive,
				Nnegative,
				Nnegative_e1,
				Nnegative_e2,
				NegativeFitted,
				NnegativeReal,
				ProbPoissonE1,
				ProbPoissonE2,
				ProbNegativeOverPositiveE1,
				ProbNegativeOverPositiveE2,
				ProbNegativeOverPositiveDif,
				ProbNegativeOverPositiveDifE1,
				ProbNegativeOverPositiveDifE2,
				ProbPoissonExpected,
				ProbPoissonExpectedE1,
				ProbPoissonExpectedE2

			]

	return output
