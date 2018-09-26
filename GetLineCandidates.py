#Change matplotlib backend to use Agg so it can run without a X server in a linux machine with Centos <7.
try:
	import matplotlib as mpl
	mpl.use('Agg')
except:
	print 'Problem using Agg backend'
# import warnings
# warnings.filterwarnings("ignore")

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#Try to import seaborn and adjust the parameters of the output plots
try:
	import seaborn as sns
	sns.set_style("white", {'legend.frameon': True})
	sns.set_style("ticks", {'legend.frameon': True})
	sns.set_context("talk")
	sns.set_palette('Dark2', 8,desat=1)
	cc = sns.color_palette()
except:
	print 'No seaborn package installed'
	cc = ['red','blue','green','orange','magenta','black']
import argparse
import astropy.io.fits as fits
from astropy import wcs
from astropy.coordinates import SkyCoord
import scipy.special
from scipy.optimize import curve_fit

'''

USAGE: "python GetLineCandidates.py -h" will give a description of the input values

python GetLineCandidates.py -Cube cube.fits -MaxSigmas 10 -MinSN 3.5 -LineSearchPath LineSearchTEST1 -SimulationPath Simulation1 -SurveyName Survey -Wavelength 3

Changelog:
---------------------------------------------------------------------------------------------
GetLineCandidates.py
Script that finds the emission lines position in the files created by 
This version works with SearchLine+v0.1.py
python GetLineCandidates.py -Cube spw1_w4.fits -LineSearchPath LineSearchTEST1 -MaxSigmas 10 -MinSN 3.5 -SimulationPath Simulation1 
---------------------------------------------------------------------------------------------

GetLineCandidates_v0.1.py
Script that finds the emission lines position in the files created by 
This version works with SearchLine.v0.1.py
This version returns all the lines with SN higher than MinSN, previously was only returning those with P<1
python GetLineCandidates_v0.1.py -Cube spw1_w4.fits -LineSearchPath LineSearchTEST1 -MaxSigmas 10 -MinSN 3.5 -SimulationPath Simulation1 
---------------------------------------------------------------------------------------------

GetLineCandidates_v0.2.py
Now it gives the output even when no Simulations are available. 
Now creates the list of candidates in the negative data.
Better handeling of the plot. 
Added two new keywords for the name of the candidates
python GetLineCandidates_v0.2.py -Cube spw1_w4.fits -LineSearchPath LineSearchTEST1 -MaxSigmas 10 -MinSN 3.5 -SimulationPath Simulation1 -SurveyName ALESS122 -Wavelength 3
---------------------------------------------------------------------------------------------

GetLineCandidates_v0.5.py
New Poisson statistics added and new outputs
I tried to estimate the purity of the sample from the simulations but they have weird behavior in this version, I need to fix it.
python GetLineCandidates_v0.5.py -Cube spw1_w4.fits -LineSearchPath LineSearchTEST1 -MaxSigmas 10 -MinSN 3.5 -SimulationPath Simulation1 -SurveyName ALESS122 -Wavelength 3
---------------------------------------------------------------------------------------------

v0.6
Updated documentation and changed the naming convention where the version will be in the header.

---------------------------------------------------------------------------------------------
v0.7
Now it gives the positions and channels of the maximum S/N pixel instead of returning the median of all the selected pixels

---------------------------------------------------------------------------------------------
v0.8
I have updated the range of points the code uses to estimate the rate of negative detections. 
Now it searches for 6 bins with number of detections higher than 20 (SN~5), this way the code tries
to better fit the tail of negative counts without being much affected by low number statistics and not going
to the low S/N regime.
I chose 6 bins because is double the degree of freedom of the function to fit (2+1). 

Modifictions to the plots.

---------------------------------------------------------------------------------------------

v0.9
I have modified the parameter EPS for DBSCAN. This parameter determines how close two detection can be from each other. 
Now the separation depends on the number of pixels per bmaj. 
If MaxSigmas is equal to 1 (Continuum images) then eps is set to 1.
---------------------------------------------------------------------------------------------


v1.0
I have added new conditions for the fit of negative counts. Now the code will better handle
the cases with low independent elements and force reasonable fits to avoid obtaining false Ppoisson=0

---------------------------------------------------------------------------------------------
'''

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
	return N*np.exp(-1.0*np.power(SNR,2)/(2.0*np.power(sigma,2)))


def get_final_SN(SourcesTotal,PixelsPerBMAJ):
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

	FinalX = []
	FinalY = []
	FinalChannel = []
	FinalSN = []
	FinalPurity = []
	for k in unique_labels:
		class_member_mask = (labels == k)
		FinalX.append(X[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalY.append(Y[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalChannel.append(Channel[class_member_mask][np.argmax(SN[class_member_mask])])
		FinalSN.append(max(SN[class_member_mask]))
		FinalPurity.append(min(purity[class_member_mask]))

	FinalX = np.array(FinalX)
	FinalY = np.array(FinalY)
	FinalChannel = np.array(FinalChannel)
	FinalSN = np.array(FinalSN)
	FinalPurity = np.array(FinalPurity)

	FinalX = FinalX[np.argsort(FinalSN)]
	FinalY = FinalY[np.argsort(FinalSN)]
	FinalChannel = FinalChannel[np.argsort(FinalSN)]
	FinalPurity = FinalPurity[np.argsort(FinalSN)]
	FinalSN = FinalSN[np.argsort(FinalSN)]
	return FinalSN



def ReadFileGetArrays(path,Sources,minSN):
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
				# COORD = np.append(COORD,np.array([source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[0][np.argmax(source[3])]]),axis=0)
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
			SN = float(j[3].replace('SN:',''))
			if SN>=minSN :
				spw = int(j[0])
				x = float(j[1])
				y = float(j[2])
				sn = float(j[-1].replace('SN:',''))
				COORD.append(np.array([x,y,spw]))
				X.append(x)
				Y.append(y)
				Channel.append(spw)
				SN_array.append(sn)
		for source in Sources:
				# COORD = np.append(COORD,np.array([source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[0][np.argmax(source[3])]]),axis=0)
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

def get_sources(files,minSN,PixelsPerBMAJ):
	files.sort()
	Sources = []
	SourcesAux = []
	# print 'Files:',files
	for i in files:
		COORD,X,Y,Channel,SN = ReadFileGetArrays(i,Sources,minSN)
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

		#check that this step works
		# Sources = np.append(Sources,SourcesAux)
		# print np.shape(Sources),np.shape(SourcesAux)
		for source in SourcesAux:
			Sources.append(source)
		# Sources = SourcesAux
	return Sources

# def GetFinalCandidates(SourcesTotalPos,PixelsPerBMAJ):
# 	COORD = []
# 	X = []
# 	Y = []
# 	Channel = []
# 	SN_array = []
# 	puritySimulation= []
# 	purityNegative = []
# 	purityPoisson = []
# 	for NewSource in SourcesTotalPos:
# 			COORD.append(np.array([NewSource[1],NewSource[2],NewSource[0]]))
# 			X.append(NewSource[1])
# 			Y.append(NewSource[2])
# 			Channel.append(NewSource[0])
# 			SN_array.append(NewSource[3])		
# 			puritySimulation.append(NewSource[5])	
# 			purityNegative.append(NewSource[6])	
# 			purityPoisson.append(NewSource[7])	

# 	COORD = np.array(COORD)
# 	X = np.array(X)
# 	Y = np.array(Y)
# 	Channel = np.array(Channel)
# 	SN = np.array(SN_array)
# 	puritySimulation = np.array(puritySimulation)
# 	purityNegative = np.array(purityNegative)
# 	purityPoisson = np.array(purityPoisson)

# 	db = DBSCAN(eps=PixelsPerBMAJ, min_samples=1,leaf_size=30).fit(COORD)
# 	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# 	core_samples_mask[db.core_sample_indices_] = True
# 	labels = db.labels_
# 	unique_labels = set(labels)


# 	# from mpl_toolkits.mplot3d import Axes3D
# 	# fig = plt.figure()
# 	# ax = fig.add_subplot(111, projection='3d')
# 	# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# 	# colors = [plt.cm.Spectral(each)
# 	#           for each in np.linspace(0, 1, len(unique_labels))]
# 	# for k, col in zip(unique_labels, colors):
# 	# 	if k == -1:
# 	# 	    # Black used for noise.
# 	# 	    col = [0, 0, 0, 1]

# 	# 	class_member_mask = (labels == k)

# 	# 	xy = COORD[class_member_mask & core_samples_mask]
# 	# 	# plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=14)
# 	# 	ax.scatter(xy[:, 0], xy[:, 1],xy[:, 2], c=tuple(col), marker='o',s=30,edgecolors='black')

# 	# 	xy = COORD[class_member_mask & ~core_samples_mask]
# 	# 	# plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
# 	# 	ax.scatter(xy[:, 0], xy[:, 1],xy[:, 2], c=tuple(col), marker='o',s=10,edgecolors='black')
# 	# 	# print xy[:, 0], xy[:, 1],xy[:, 2]

# 	# plt.title('X,Y: %d' % n_clusters_)
# 	# ax.set_xlabel('X')
# 	# ax.set_ylabel('Y')
# 	# ax.set_zlabel('Channel')
# 	# plt.show()



# 	FinalX = []
# 	FinalY = []
# 	FinalChannel = []
# 	FinalSN = []
# 	FinalPuritySimulation = []
# 	FinalPurityNegative = []
# 	FinalPurityPoisson = []
# 	for k in unique_labels:
# 		class_member_mask = (labels == k)
# 		FinalX.append(X[class_member_mask][np.argmax(SN[class_member_mask])])
# 		FinalY.append(Y[class_member_mask][np.argmax(SN[class_member_mask])])
# 		FinalChannel.append(Channel[class_member_mask][np.argmax(SN[class_member_mask])])
# 		FinalSN.append(max(SN[class_member_mask]))
# 		FinalPuritySimulation.append(min(puritySimulation[class_member_mask]))
# 		FinalPurityNegative.append(min(min(purityNegative[class_member_mask]),1))
# 		FinalPurityPoisson.append(min(purityPoisson[class_member_mask]))

# 	FinalX = np.array(FinalX)
# 	FinalY = np.array(FinalY)
# 	FinalChannel = np.array(FinalChannel)
# 	FinalSN = np.array(FinalSN)
# 	FinalPuritySimulation = np.array(FinalPuritySimulation)
# 	FinalPurityNegative = np.array(FinalPurityNegative)
# 	FinalPurityPoisson = np.array(FinalPurityPoisson)

# 	FinalX = FinalX[np.argsort(FinalSN)]
# 	FinalY = FinalY[np.argsort(FinalSN)]
# 	FinalChannel = FinalChannel[np.argsort(FinalSN)]
# 	FinalPuritySimulation = FinalPuritySimulation[np.argsort(FinalSN)]
# 	FinalPurityNegative = FinalPurityNegative[np.argsort(FinalSN)]
# 	FinalPurityPoisson = FinalPurityPoisson[np.argsort(FinalSN)]
# 	FinalSN = FinalSN[np.argsort(FinalSN)]
# 	return FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN

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


	FinalSN = FinalSN[np.argsort(FinalSN)]
	return FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN,FinalPSimultionE1,FinalPSimultionE2,FinalPPoissonE1,FinalPPoissonE2


def GetPoissonEstimates(bins,SNFinalPos,SNFinalNeg):
	ProbPoisson = []
	ProbPoissonE1 = []
	ProbPoissonE2 = []
	ProbNegativeOverPositive = []
	PurityPoisson = []
	Nnegative = []
	NnegativeReal = []
	NPositive = []
	Nnegative_e1 = []
	Nnegative_e2 = []
	for sn in bins:
		if len(SNFinalPos[SNFinalPos>=sn])>0:
			ProbNegativeOverPositive.append(len(SNFinalNeg[SNFinalNeg>=sn])*1.0/len(SNFinalPos[SNFinalPos>=sn]))
		elif len(SNFinalNeg[SNFinalNeg>=sn])>0:
			ProbNegativeOverPositive.append(1.0)
		else:
			ProbNegativeOverPositive.append(0.0)
		k = len(SNFinalNeg[SNFinalNeg>=sn])
		aux = scipy.special.gammaincinv(k + 1, [0.16,0.5,0.84])
		NnegativeReal.append(k)
		Nnegative.append(aux[1])
		Nnegative_e1.append(aux[1]-aux[0])
		Nnegative_e2.append(aux[2]-aux[1])
		NPositive.append(1.0*len(SNFinalPos[SNFinalPos>=sn]))

	Nnegative = np.array(Nnegative)
	NnegativeReal = np.array(NnegativeReal)
	Nnegative_e1 = np.array(Nnegative_e1)
	Nnegative_e2 = np.array(Nnegative_e2)
	

	MinSNtoFit = min(bins)
	UsableBins = len(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>args.LimitN])
	print 'Min SN to do the fit:',round(MinSNtoFit,1),', Number of usable bins:',UsableBins
	if UsableBins<6:
		print '*** We are using ',UsableBins,' points for the fitting of the negative counts ***'
		print '*** We usually get good results with 6 points, try reducing the parameter -MinSN ***'
	while UsableBins>6:
		MinSNtoFit = MinSNtoFit + 0.1
		UsableBins = len(Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>args.LimitN])
		print 'Min SN to do the fit:',round(MinSNtoFit,1),', Number of usable bins:',UsableBins
		if MinSNtoFit>max(bins):
			print 'No negative points to do the fit'
			exit()


	if UsableBins>=3:
		try:
			popt, pcov = curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>args.LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>args.LimitN],p0=[1e6,1])
			perr = np.sqrt(np.diag(pcov))
			# print popt,popt/perr,not np.isfinite(perr[0])
			CounterFitTries = 0
			while not np.isfinite(perr[0]):
				print '*** curve_fit failed to converge ... ***'
				NewParameter1 = np.power(10,np.random.uniform(1,9))
				NewParameter2 = np.random.uniform(0.1,2.0)
				print '*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***'
				popt, pcov = curve_fit(NegativeRate, bins[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>args.LimitN], Nnegative[bins>=MinSNtoFit][Nnegative[bins>=MinSNtoFit]>args.LimitN],p0=[NewParameter1,NewParameter2])
				perr = np.sqrt(np.diag(pcov))
				print '*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***'
				CounterFitTries += 1
				if CounterFitTries >100:
					print '*** Over 100 attemps and not good fit *** '
					break

		except:
			print 'Fitting failed for LimitN:'+str(args.LimitN)+' and '+str(args.MinSN)+'... Will force LimitN=0'
			popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[1e6,1])	
			perr = np.sqrt(np.diag(pcov))
			# print popt,popt/perr,not np.isfinite(perr[0])
			CounterFitTries = 0
			while not np.isfinite(perr[0]):
				print '*** curve_fit failed to converge ... ***'
				NewParameter1 = np.power(10,np.random.uniform(1,9))
				NewParameter2 = np.random.uniform(0.1,2.0)
				print '*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***'
				popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[NewParameter1,NewParameter2])
				perr = np.sqrt(np.diag(pcov))
				print '*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***'
				CounterFitTries += 1
				if CounterFitTries >100:
					print '*** Over 100 attemps and not good fit *** '
					break
	else:
		print 'Number of usable bins is less than 3 for LimitN:'+str(args.LimitN)+' and '+str(args.MinSN)+'... Will force LimitN=0'
		popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[1e6,1])	
		perr = np.sqrt(np.diag(pcov))
		# print popt,popt/perr,not np.isfinite(perr[0])
		CounterFitTries = 0
		while not np.isfinite(perr[0]):
			print '*** curve_fit failed to converge ... ***'
			NewParameter1 = np.power(10,np.random.uniform(1,9))
			NewParameter2 = np.random.uniform(0.1,2.0)
			print '*** New Initial Estimates for the fitting (random):',round(NewParameter1),round(NewParameter2,2),' ***'
			popt, pcov = curve_fit(NegativeRate, bins[Nnegative>0], Nnegative[Nnegative>0],p0=[NewParameter1,NewParameter2])
			perr = np.sqrt(np.diag(pcov))
			print '*** New Results: N:',round(popt[0]),' +/- ',round(perr[0]),' Sigma:',round(popt[1],2),' +/- ',round(perr[1],2),' ***'
			CounterFitTries += 1
			if CounterFitTries >100:
				print '*** Over 100 attemps and not good fit *** '
				break

	NegativeFitted = NegativeRate(bins,popt[0],popt[1])
	SNPeakGaussian = (popt/np.sqrt(np.diag(pcov)))[0]

	for i in range(len(bins)):
		aux = []
		for j in range(1000):
			# print np.random.normal(NegativeFitted[i],NegativeFitted[i]/SNPeakGaussian)
			aux.append(1-scipy.special.gammaincc(0+1,np.random.normal(NegativeFitted[i],NegativeFitted[i]/SNPeakGaussian)))
		PP = np.nanpercentile(aux,[16,50,84])
		# print bins[i],round(PP[1]*100.0,1),' - ',round((PP[1]-PP[0])*100.0,1),'+',round((PP[2]-PP[1])*100.0,1)

		# ProbPoisson.append(1-scipy.special.gammaincc(0+1,NegativeFitted[i]))
		ProbPoisson.append(PP[1])
		ProbPoissonE1.append(PP[1]-PP[0])
		ProbPoissonE2.append(PP[2]-PP[1])
		if NPositive[i]>0:
			PurityPoisson.append(max((NPositive[i]-NegativeFitted[i])/NPositive[i],0))
		else:
			PurityPoisson.append(0.0)

	ProbPoisson = np.array(ProbPoisson)
	ProbPoissonE1 = np.array(ProbPoissonE1)
	ProbPoissonE2 = np.array(ProbPoissonE2)
	ProbNegativeOverPositive = np.array(ProbNegativeOverPositive)
	PurityPoisson = np.array(PurityPoisson)

	return bins,ProbPoisson,ProbNegativeOverPositive,PurityPoisson,NPositive,Nnegative,Nnegative_e1,Nnegative_e2,NegativeFitted,NnegativeReal,ProbPoissonE1,ProbPoissonE2




parser = argparse.ArgumentParser(description="Python script that finds line emission-like features in an ALMA data cube")
parser.add_argument('-Cube', type=str, required=True,help = 'Path to the Cube fits file where the search will be done')
parser.add_argument('-LineSearchPath', type=str, default='OutputLineSearch', required=False , help = 'Directory where the outputs will be saved [Default:LineSearchPath]')
parser.add_argument('-SimulationPath', type=str, default='Simulation', required=False , help = 'Directory where the simulations should be found [Default:Simulation]')
parser.add_argument('-MaxSigmas', type=int, default = 10, required=False,help = 'Maximum number of channels to use as sigma value for the spectral Gaussian convolution. [Default:10]')
parser.add_argument('-MinSN', type=float, default = 5.0, required=False,help = 'Minimum S/N value to save in the outputs. A good value depends on each data cube, reasonable values are bettween 3.5 and 6 [Default:5.0]')
parser.add_argument('-SurveyName', type=str, default='Survey', required=False , help = 'Name to identify the line candidates [Default:Survey]')
parser.add_argument('-Wavelength', type=str, default='X', required=False , help = 'Wavelength for reference in the names [Default:X]')
parser.add_argument('-LimitN', type=float, default='20.0', required=False , help = 'Limit for the number of detection above certain S/N to be used in the fitting of the negative counts [Default:20]')
parser.add_argument('-LegendFontSize', type=float, default='10.0', required=False , help = 'Fontsize fot the figures legends [Default:10]')
parser.add_argument('-UserEPS', type=str, default='False',choices=['True','False'], required=False , help = 'Whether to use EPS value entered from user otherwise use number of pixels per bmaj [Default:False]')
parser.add_argument('-EPS', type=float, default=5.0, required=False , help = 'EPS value to use if User sets -UserEPS to True [Default:5.0]')
parser.add_argument('-GetTotalEstimate', type=str, default='False',choices=['True','False'], required=False , help = 'Whether to get total distribution including all lines of different widths [Default:True]')

args = parser.parse_args()

if args.UserEPS=='False':

	PixelsPerBMAJ = GetPixelsPerBMAJ(args.Cube)

	# PixelsPerBMAJ = 1.0
	if args.MaxSigmas == 1:
		PixelsPerBMAJ = 1.0
	print '*** Using EPS value of '+str(PixelsPerBMAJ)+'***'
else:
	PixelsPerBMAJ = args.EPS
	print '*** Using EPS value of '+str(PixelsPerBMAJ)+'***'


# w, h = 1.0*plt.figaspect(0.9)
# fig = plt.figure(figsize=(w,h))
# plt.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
# ax1 = plt.subplot(111)


SourcesTotalPos = []
SourcesTotalNeg = []

w, h = 1.0*plt.figaspect(0.9)
fig1 = plt.figure(figsize=(w,h))
fig1.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
ax1 = fig1.add_subplot(111)

for i in range(args.MaxSigmas):
	print 50*'-'
	Sources_real = np.array(get_sources([args.LineSearchPath+'/line_dandidates_sn_sigmas'+str(i)+'_pos'],args.MinSN,PixelsPerBMAJ))
	Sources_realNeg = np.array(get_sources([args.LineSearchPath+'/line_dandidates_sn_sigmas'+str(i)+'_neg'],args.MinSN,PixelsPerBMAJ))

	simulations_folders = glob.glob(args.SimulationPath+'/simul_*')
	SimulatedSources = []
	for folder in simulations_folders:
		try:
			aux = get_sources([folder+'/line_dandidates_sn_sigmas'+str(i)+'_pos'],args.MinSN,PixelsPerBMAJ)
			aux_sn = []
			for source in aux:
				aux_sn.append(max(source[3]))
			aux_sn = np.array(aux_sn)
			SimulatedSources.append(aux_sn)

			aux = get_sources([folder+'/line_dandidates_sn_sigmas'+str(i)+'_neg'],args.MinSN,PixelsPerBMAJ)
			aux_sn = []
			for source in aux:
				aux_sn.append(max(source[3]))
			aux_sn = np.array(aux_sn)
			SimulatedSources.append(aux_sn)
		except:
			print 'file not working',folder+'/line_dandidates_sn_sigmas'+str(i)+'_pos'

	SNReal = []
	for source in Sources_real:
		SNReal.append(max(source[3]))
	SNReal = np.array(SNReal)

	SNRealNeg = []
	for source in Sources_realNeg:
		SNRealNeg.append(max(source[3]))
	SNRealNeg = np.array(SNRealNeg)

	SimulatedSources = np.array(SimulatedSources)

	bins = np.arange(args.MinSN,7.1,0.1)
	print 'for sigma',i
	y = []
	print 'S/N NDetected Fraction Nsimulations'
	if len(SimulatedSources)>1:
		for sn in bins:

			print round(sn,1),len(SNReal[SNReal>=sn])*1.0,
			N_simulations2 = 1.0*len(SimulatedSources)
			N_detections = 0.0
			for sim in SimulatedSources:
				if len(sim[sim>=sn])>0:
					N_detections += 1.0
			print round(N_detections/N_simulations2,2),N_simulations2
			y.append(N_detections/N_simulations2)
	else:
		y = np.zeros_like(bins)
		N_simulations2 = 0.0


	if i<7:
		ax1.plot(bins,y,'-',label=r' $\sigma$ = '+str(i)+' channels')
	if i>=7 and i<14:
		ax1.plot(bins,y,'--',label=r' $\sigma$ = '+str(i)+' channels')
	if i>=14:
		ax1.plot(bins,y,':',label=r' $\sigma$ = '+str(i)+' channels')



	bins,ProbPoisson,ProbNegativeOverPositive,PurityPoisson,NPositive,Nnegative,Nnegative_e1,Nnegative_e2,NegativeFitted,NnegativeReal,ProbPoissonE1,ProbPoissonE2 = GetPoissonEstimates(bins,SNReal,SNRealNeg)

	w, h = 1.0*plt.figaspect(0.9)
	fig2 = plt.figure(figsize=(w,h))
	fig2.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
	ax2 = fig2.add_subplot(111)
	ax2.semilogy(bins,NPositive,'-',color=cc[0],label='Positive Detections')
	ax2.errorbar(bins[NnegativeReal>0],Nnegative[NnegativeReal>0],yerr=[Nnegative_e1[NnegativeReal>0],Nnegative_e2[NnegativeReal>0]],fmt='o',color=cc[1],label='Negative Detections for sigmas:'+str(i))
	ax2.semilogy(bins,NegativeFitted,'-',color=cc[2],label='Fitted negative underlying rate')
	ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('N',fontsize=20)
	if len(bins[NnegativeReal>0])>0:
		ax2.legend(loc=0,fontsize=args.LegendFontSize,ncol=1)
	ax2.tick_params(axis='both', which='major', labelsize=20)
	ax2.set_ylim(ymin=0.1)
	ax2.set_xticks(np.arange(int(args.MinSN),8,1))
	ax2.grid(True)
	fig2.savefig('NumberPositiveNegative_'+str(i)+'.pdf')
	for source in Sources_real:
		if max(source[3])>=args.MinSN:


			k = np.interp(source[3][np.argmax(source[3])],bins,y)*N_simulations2
			n = N_simulations2
			aux = scipy.special.betaincinv(k+1, n+1-k, [0.16,0.5,0.84])
			ErrorPSimulation_1 = aux[1]-aux[0]
			ErrorPSimulation_2 = aux[2]-aux[1]			
			if N_simulations2==0:
				ErrorPSimulation_1 = 0.0
				ErrorPSimulation_2 = 0.0
			index = np.argmin(abs(bins - max(source[3])))
			# print max(source[3]),bins[index],index
			ErrorPPoisson_1 = ProbPoissonE1[index]
			ErrorPPoisson_2 = ProbPoissonE2[index]
			# if max(source[3])>=6.0:
				# print max(source[3]),ErrorPSimulation_1,ErrorPSimulation_2,ErrorPPoisson_1,ErrorPPoisson_2
			NewSource = [source[0][np.argmax(source[3])],source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[3][np.argmax(source[3])],source[4],np.interp(source[3][np.argmax(source[3])],bins,y),np.interp(source[3][np.argmax(source[3])],bins,ProbNegativeOverPositive),np.interp(source[3][np.argmax(source[3])],bins,ProbPoisson),ErrorPSimulation_1,ErrorPSimulation_2,ErrorPPoisson_1,ErrorPPoisson_2]
			SourcesTotalPos.append(NewSource)

	for source in Sources_realNeg:
		if max(source[3])>=args.MinSN:
			k = np.interp(source[3][np.argmax(source[3])],bins,y)*N_simulations2
			n = N_simulations2
			aux = scipy.special.betaincinv(k+1, n+1-k, [0.16,0.5,0.84])
			ErrorPSimulation_1 = aux[1]-aux[0]
			ErrorPSimulation_2 = aux[2]-aux[1]			
			if N_simulations2==0:
				ErrorPSimulation_1 = 0.0
				ErrorPSimulation_2 = 0.0

			index = np.argmin(abs(bins - max(source[3])))
			# print source[3],bins,bins[i]
			ErrorPPoisson_1 = ProbPoissonE1[index]
			ErrorPPoisson_2 = ProbPoissonE2[index]			
			NewSource = [source[0][np.argmax(source[3])],source[1][np.argmax(source[3])],source[2][np.argmax(source[3])],source[3][np.argmax(source[3])],source[4],np.interp(source[3][np.argmax(source[3])],bins,y),np.interp(source[3][np.argmax(source[3])],bins,ProbNegativeOverPositive),np.interp(source[3][np.argmax(source[3])],bins,ProbPoisson),ErrorPSimulation_1,ErrorPSimulation_2,ErrorPPoisson_1,ErrorPPoisson_2]
			SourcesTotalNeg.append(NewSource)

SNFinalPos = get_final_SN(SourcesTotalPos,PixelsPerBMAJ)
SNFinalNeg = get_final_SN(SourcesTotalNeg,PixelsPerBMAJ)

bins = np.arange(args.MinSN,7.1,0.1)


if args.GetTotalEstimate == 'True':
	print '*** Creating output from total estimates... ***'
	##### For Total #####
	print 'Reading Simulations for Total estimate...'
	simulations_folders = glob.glob(args.SimulationPath+'/simul_*')
	SimulatedSourcesTotal = []
	counter = 1
	for folder in simulations_folders:
		print folder,counter,'/',len(simulations_folders)
		try:
			aux = get_sources(glob.glob(folder+'/*_pos.*'),args.MinSN,PixelsPerBMAJ)

			aux_sn = []
			for source in aux:
				aux_sn.append(max(source[3]))
			aux_sn = np.array(aux_sn)
			SimulatedSourcesTotal.append(aux_sn)

			aux = get_sources(glob.glob(folder+'/*_neg.*'),args.MinSN,PixelsPerBMAJ)

			aux_sn = []
			for source in aux:
				aux_sn.append(max(source[3]))
			aux_sn = np.array(aux_sn)
			SimulatedSourcesTotal.append(aux_sn)

		except:
			print 'file not working',folder
		counter += 1
	SimulatedSourcesTotal = np.array(SimulatedSourcesTotal)
	yTotal = []
	NSimulations = []
	for sn in bins:
		N_simulations2 = 1.0*len(SimulatedSourcesTotal)
		N_detections = 0.0
		aux = []
		for sim in SimulatedSourcesTotal:
			if len(sim[sim>=sn])>0:
				N_detections += 1.0
			aux.append(len(sim[sim>=sn]))
		NSimulations.append(np.median(aux))
		if N_simulations2>0:
			yTotal.append(N_detections/N_simulations2)
		else:
			yTotal.append(-1.0)
	NSimulations = np.array(NSimulations)

	yTotal[yTotal>1] = 1
	ax1.plot(bins,yTotal,'--',color='green',label='Simulations Total',lw=3)

	# bins,ProbPoisson,ProbNegativeOverPositive,PurityPoisson,NPositive,Nnegative,Nnegative_e1,Nnegative_e2,NegativeFitted,NnegativeReal = GetPoissonEstimates(bins,SNFinalPos,SNFinalNeg)
	bins,ProbPoisson,ProbNegativeOverPositive,PurityPoisson,NPositive,Nnegative,Nnegative_e1,Nnegative_e2,NegativeFitted,NnegativeReal,ProbPoissonE1,ProbPoissonE2 = GetPoissonEstimates(bins,SNFinalPos,SNFinalNeg)
	ProbNegativeOverPositive[ProbNegativeOverPositive>1] = 1
	ProbPoisson[ProbPoisson>1] = 1
	ax1.plot(bins,ProbNegativeOverPositive,'--',color='black',label='#Neg[>=sn]/#Pos[>=sn]',lw=3)
	ax1.plot(bins,ProbPoisson,'--',color='red',label='Poisson',lw=3)
	ax1.set_xlabel('S/N',fontsize=20)
	ax1.set_ylabel('Probability produced by noise ',fontsize=15)
	if args.MaxSigmas<20:
		ax1.legend(loc='best',fontsize=args.LegendFontSize,ncol=1)
	else:
		if args.MaxSigmas<40:
			ax1.legend(loc='best',fontsize=args.LegendFontSize,ncol=2)
		else:
			ax1.legend(loc='best',fontsize=4,ncol=2)
	ax1.tick_params(axis='both', which='major', labelsize=20)
	ax1.set_ylim(-0.1,1.1)
	fig1.savefig('ProbabilityFalseSN.pdf')


	w, h = 1.0*plt.figaspect(0.9)
	fig = plt.figure(figsize=(w,h))
	plt.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
	ax1 = plt.subplot(111)
	plt.semilogy(bins,NPositive,'-',color=cc[0],label='Positive Detections')
	plt.errorbar(bins[NnegativeReal>0],Nnegative[NnegativeReal>0],yerr=[Nnegative_e1[NnegativeReal>0],Nnegative_e2[NnegativeReal>0]],fmt='o',color=cc[1],label='Negative Detections')
	plt.semilogy(bins,NegativeFitted,'-',color=cc[2],label='Fitted negative underlying rate')
	plt.xlabel('S/N',fontsize=20)
	plt.ylabel('N',fontsize=20)
	plt.legend(loc=0,fontsize=args.LegendFontSize,ncol=1)
	plt.tick_params(axis='both', which='major', labelsize=20)
	plt.ylim(ymin=0.1)
	plt.grid(True)
	plt.xticks(np.arange(int(args.MinSN),8,1))

	plt.savefig('NumberPositiveNegativeTotal.pdf')

	w, h = 1.0*plt.figaspect(0.9)
	fig = plt.figure(figsize=(w,h))
	plt.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
	ax1 = plt.subplot(111)
	AuxPuritySimulation = (NPositive-NSimulations)/NPositive
	AuxPurityNegatives = (NPositive-NnegativeReal)/NPositive
	AuxPuritySimulation[AuxPuritySimulation<0] = 0
	AuxPurityNegatives[AuxPurityNegatives<0] = 0
	PurityPoisson[PurityPoisson<0] = 0
	plt.plot(bins,AuxPuritySimulation,'-',color='green',label='Simulations Total',lw=3)
	plt.plot(bins,AuxPurityNegatives,'-',color='black',label='#(Pos[>=sn] - Neg[>=sn])/#Pos[>=sn]',lw=3)
	plt.plot(bins,PurityPoisson,'-',color='red',label='Poisson',lw=3)
	plt.xlabel('S/N',fontsize=20)
	plt.ylabel('Purity',fontsize=20)
	plt.legend(loc=0,fontsize=args.LegendFontSize,ncol=1)
	plt.tick_params(axis='both', which='major', labelsize=20)
	plt.ylim(-0.1,1.1)
	plt.savefig('Purity.pdf')

	################################################
	####  This part is just to write the file ######
	Output = open('PuritySample.dat','w')
	Output.write('#S/N PuritySimulations PurityNegative PurityPoisson\n')
	for i in range(len(bins)):
		if NPositive[i]>0:
			Output.write(str(bins[i])+' '+str(max(round((NPositive[i]-NSimulations[i])/NPositive[i],2),0))+' '+str(max(round((NPositive[i]-NnegativeReal[i])/NPositive[i],2),0))+' '+str(round(PurityPoisson[i],2))+'\n')
		else:
			Output.write(str(bins[i])+' 0 '+str(round(PurityPoisson[i],2))+'\n')

	Output.close()
	################################################



	################################################
	####  This part is just to write the file ######
	Output = open('ProbabilityFalse.dat','w')
	Output.write('#S/N ProbSimulationTotal ProbNegative ProbPoisson\n')
	# print 'SN neg:'
	for i in range(len(bins)):
		Output.write(str(bins[i])+' '+str(round(yTotal[i],2))+' '+str(round(ProbNegativeOverPositive[i],2))+' '+str(round(ProbPoisson[i],2))+'\n')

	Output.close()
	################################################
else:
	print '*** No output from total estimates... ***'
	if args.MaxSigmas == 1:
		print '*** Since MaxSigmas is 1 you should consider turning on the option for GetTotalEstimate ... ***'

######## Positives ########

# FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN = GetFinalCandidates(SourcesTotalPos,PixelsPerBMAJ)
FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN,FinalPSimultionE1,FinalPSimultionE2,FinalPPoissonE1,FinalPPoissonE2= GetFinalCandidates(SourcesTotalPos,PixelsPerBMAJ)

hdulist =   fits.open(args.Cube,memmap=True)
w = wcs.WCS(hdulist[0].header)

[ra,dec,freq,stoke] =  w.all_pix2world(FinalX,FinalY,FinalChannel,np.zeros_like(FinalChannel),0)
c = []
for i in range(len(ra)):
  c.append(SkyCoord(ra[i], dec[i], frame='icrs', unit='deg'))

Output = open('LineCandidatesPositive.dat','w')
Output.write('#ID RA DEC Frequency S/N ProbabilityFalseSimulation ProbabilityFalseSimulationE1 ProbabilityFalseSimulationE2 ProbabilityFalseNegative ProbabilityFalsePoisson ProbabilityFalsePoissonE1 ProbabilityFalsePoissonE2\n')

for i in range(len(FinalX)):
  k = i + 1
  i = len(FinalX)-i-1
  # Output.write(args.SurveyName+'-'+args.Wavelength+'mm.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '+str(round(FinalPuritySimulation[i],2))+' '+str(round(FinalPurityNegative[i],2))+' '+str(round(FinalPurityPoisson[i],2))+'\n')
  # Output.write(args.SurveyName+'-'+args.Wavelength+'mm.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '+str(round(FinalPuritySimulation[i],2))+' '+str(round(FinalPurityNegative[i],2))+' '+format(FinalPurityPoisson[i],'.1e')+'\n')
  # Output.write(args.SurveyName+'-'+args.Wavelength+'mm.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '+str(round(FinalPuritySimulation[i],2)) +' -'+str(round(FinalPSimultionE1[i],2))+' +'+str(round(FinalPSimultionE2[i],2))+' '+str(round(FinalPurityNegative[i],2))+' '+format(FinalPurityPoisson[i],'.1e')+' -'+format(FinalPPoissonE1[i],'.1e')+' +'+format(FinalPPoissonE2[i],'.1e')+'\n')
  Output.write(args.SurveyName+'-'+args.Wavelength+'mm.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '+format(FinalPuritySimulation[i],'.1e') +' -'+format(FinalPSimultionE1[i],'.1e')+' +'+format(FinalPSimultionE2[i],'.1e')+' '+str(round(FinalPurityNegative[i],2))+' '+format(FinalPurityPoisson[i],'.1e')+' -'+format(FinalPPoissonE1[i],'.1e')+' +'+format(FinalPPoissonE2[i],'.1e')+'\n')

Output.close()


######## Negatives ########
# FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN = GetFinalCandidates(SourcesTotalNeg,PixelsPerBMAJ)
FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN,FinalPSimultionE1,FinalPSimultionE2,FinalPPoissonE1,FinalPPoissonE2= GetFinalCandidates(SourcesTotalNeg,PixelsPerBMAJ)


[ra,dec,freq,stoke] =  w.all_pix2world(FinalX,FinalY,FinalChannel,np.zeros_like(FinalChannel),0)
c = []
for i in range(len(ra)):
  c.append(SkyCoord(ra[i], dec[i], frame='icrs', unit='deg'))

Output = open('LineCandidatesNegative.dat','w')
Output.write('#ID RA DEC Frequency S/N ProbabilityFalseSimulation ProbabilityFalseSimulationE1 ProbabilityFalseSimulationE2 ProbabilityFalseNegative ProbabilityFalsePoisson ProbabilityFalsePoissonE1 ProbabilityFalsePoissonE2\n')

for i in range(len(FinalX)):
  k = i + 1
  i = len(FinalX)-i-1
  # Output.write(args.SurveyName+'-'+args.Wavelength+'mm.NEG.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '+str(round(FinalPuritySimulation[i],2))+' '+str(round(FinalPurityNegative[i],2))+' '+format(FinalPurityPoisson[i],'.1e')+'\n')
  # Output.write(args.SurveyName+'-'+args.Wavelength+'mm.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '+str(round(FinalPuritySimulation[i],2)) +' -'+str(round(FinalPSimultionE1[i],2))+' +'+str(round(FinalPSimultionE2[i],2))+' '+str(round(FinalPurityNegative[i],2))+' '+format(FinalPurityPoisson[i],'.1e')+' -'+format(FinalPPoissonE1[i],'.1e')+' +'+format(FinalPPoissonE2[i],'.1e')+'\n')
  Output.write(args.SurveyName+'-'+args.Wavelength+'mm.Neg.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '+format(FinalPuritySimulation[i],'.1e') +' -'+format(FinalPSimultionE1[i],'.1e')+' +'+format(FinalPSimultionE2[i],'.1e')+' '+str(round(FinalPurityNegative[i],2))+' '+format(FinalPurityPoisson[i],'.1e')+' -'+format(FinalPPoissonE1[i],'.1e')+' +'+format(FinalPPoissonE2[i],'.1e')+'\n')


Output.close()

print 'EPS used in DBSCAN:',PixelsPerBMAJ

