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
import LineSeekerFunctions
import os.path

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

v1.1
I moved the functions to the new file LineSeekerFunctions. 
New fidelity estimates that give the probability of any line of being more than noise. 

---------------------------------------------------------------------------------------------
'''


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

print 20*'#','Checking inputs....',20*'#'
if os.path.exists(args.Cube):
    print '*** Cube',args.Cube,'found ***'
else:
    print '*** Cube',args.Cube,'not found ***\naborting..'
    exit()

if args.MaxSigmas<1:
    print '*** The value for MaxSigmas of',args.MaxSigmas,'is too small ***\naborting..'
    exit()
else:
    print '*** The value for MaxSigmas of',args.MaxSigmas,'is ok ***'

if args.MinSN<0:
    print '*** The value for MinSN of',args.MinSN,'has to be positive ***\naborting..'
    exit()
else:
    print '*** The value for MinSN of',args.MinSN,'is ok ***'

if args.UserEPS=='False':
	PixelsPerBMAJ = LineSeekerFunctions.GetPixelsPerBMAJ(args.Cube)
	if args.MaxSigmas == 1:
		PixelsPerBMAJ = 1.0
	print '*** Using EPS value of '+str(PixelsPerBMAJ)+'***'
else:
	PixelsPerBMAJ = args.EPS
	print '*** Using EPS value of '+str(PixelsPerBMAJ)+'***'


bins = np.arange(args.MinSN,8.1,0.1)
SourcesTotalPos = []
SourcesTotalNeg = []

w, h = 1.0*plt.figaspect(0.9)
fig1 = plt.figure(figsize=(w,h))
fig1.subplots_adjust(left=0.15, bottom=0.13, right=0.94, top=0.96,wspace=0.10, hspace=0.2)
ax1 = fig1.add_subplot(111)

for i in range(args.MaxSigmas):
	print 50*'-'
	Sources_real = np.array(LineSeekerFunctions.GetSourcesFromFiles([args.LineSearchPath+'/line_dandidates_sn_sigmas'+str(i)+'_pos'],args.MinSN,PixelsPerBMAJ))
	Sources_realNeg = np.array(LineSeekerFunctions.GetSourcesFromFiles([args.LineSearchPath+'/line_dandidates_sn_sigmas'+str(i)+'_neg'],args.MinSN,PixelsPerBMAJ))

	simulations_folders = glob.glob(args.SimulationPath+'/simul_*')
	SimulatedSources = []

	for folder in simulations_folders:
		try:
			aux = LineSeekerFunctions.GetSourcesFromFiles([folder+'/line_dandidates_sn_sigmas'+str(i)+'_pos'],args.MinSN,PixelsPerBMAJ)
			aux_sn = []

			for source in aux:
				aux_sn.append(max(source[3]))

			aux_sn = np.array(aux_sn)
			SimulatedSources.append(aux_sn)

			aux = LineSeekerFunctions.GetSourcesFromFiles([folder+'/line_dandidates_sn_sigmas'+str(i)+'_neg'],args.MinSN,PixelsPerBMAJ)
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

	
	print 'for sigma',i
	y = []
	yExpected = []
	yExpectedSigma = []
	print 'S/N NDetected Fraction Nsimulations ExpectedNumberPerCube Error'
	if len(SimulatedSources)>1:
		for sn in bins:
			# print round(sn,1),len(SNReal[SNReal>=sn])*1.0,
			N_simulations2 = 1.0*len(SimulatedSources)
			N_detections = 0.0
			NumberDetectedPerCube = []

			for sim in SimulatedSources:
				NumberDetectedPerCube.append(len(sim[sim>=sn])*1.0)
				if len(sim[sim>=sn])>0:
					N_detections += 1.0

			print round(sn,1),
			print len(SNReal[SNReal>=sn]),
			print round(N_detections/N_simulations2,2),
			print N_simulations2,
			print round(np.mean(NumberDetectedPerCube),3),
			print round(np.std(NumberDetectedPerCube)/np.sqrt(1.0*len(NumberDetectedPerCube)),3)

			y.append(N_detections/N_simulations2)
			yExpected.append(np.mean(NumberDetectedPerCube))
			yExpectedSigma.append(np.std(NumberDetectedPerCube)/np.sqrt(len(NumberDetectedPerCube)))
	else:
		y = np.zeros_like(bins)
		yExpected = np.zeros_like(bins)
		N_simulations2 = 0.0

	y = np.array(y)
	yExpected = np.array(yExpected)
	yExpectedSigma = np.array(yExpectedSigma)

	if i<7:
		ax1.plot(bins,y,'-',label=r' $\sigma$ = '+str(i)+' channels')
	if i>=7 and i<14:
		ax1.plot(bins,y,'--',label=r' $\sigma$ = '+str(i)+' channels')
	if i>=14:
		ax1.plot(bins,y,':',label=r' $\sigma$ = '+str(i)+' channels')


	Sources_real = Sources_real[np.argsort(SNReal)][::-1]
	SNReal = SNReal[np.argsort(SNReal)][::-1]
	Sources_realNeg = Sources_realNeg[np.argsort(SNRealNeg)][::-1]
	SNRealNeg = SNRealNeg[np.argsort(SNRealNeg)][::-1]
	bins,ProbPoisson,ProbNegativeOverPositive,PurityPoisson,NPositive,Nnegative,Nnegative_e1,Nnegative_e2,NegativeFitted,NnegativeReal,ProbPoissonE1,ProbPoissonE2,ProbNegativeOverPositiveE1,ProbNegativeOverPositiveE2,ProbNegativeOverPositiveDif,ProbNegativeOverPositiveDifE1,ProbNegativeOverPositiveDifE2,ProbPoissonExpected,ProbPoissonExpectedE1,ProbPoissonExpectedE2 = LineSeekerFunctions.GetPoissonEstimates(bins,SNReal,SNRealNeg,args.LimitN,args.MinSN)

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
			sn = max(source[3])

			aux,ErrorPSimulation_1,ErrorPSimulation_2 = LineSeekerFunctions.GetPoissonErrorGivenMeasurements(np.interp(sn,bins,y)*N_simulations2,N_simulations2)
			if N_simulations2==0:
				ErrorPSimulation_1 = 0.0
				ErrorPSimulation_2 = 0.0

			index = np.argmin(abs(bins - sn))
			ErrorPPoisson_1 = ProbPoissonE1[index]
			ErrorPPoisson_2 = ProbPoissonE2[index]

			PNegativeTotalE1Real = np.interp(sn,bins,ProbNegativeOverPositiveE1)
			PNegativeTotalE2Real = np.interp(sn,bins,ProbNegativeOverPositiveE2)
			PNegativeDifTotalReal = np.interp(sn,bins,ProbNegativeOverPositiveDif)
			PNegativeDifTotalE1Real = np.interp(sn,bins,ProbNegativeOverPositiveDifE1)
			PNegativeDifTotalE2Real = np.interp(sn,bins,ProbNegativeOverPositiveDifE2)

			if N_simulations2>0:
				Rate = np.interp(sn,bins,yExpected) - np.interp(sn+0.1,bins,yExpected)
				RateSigma = 0.5*np.sqrt(np.sum(np.power(np.array([np.interp(sn,bins,yExpectedSigma),np.interp(sn+0.1,bins,yExpectedSigma)]),2)))
				auxSimulationExpected = [Rate - RateSigma,Rate,Rate + RateSigma]
				Psimexpected = 1.0 - max(0,len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]) - auxSimulationExpected[1])*1.0/(len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]))
				Psimexpected1 = 1.0 - max(0,len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]) - auxSimulationExpected[0])*1.0/(len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]))
				Psimexpected2 = 1.0 - max(0,len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]) - auxSimulationExpected[2])*1.0/(len(SNReal[SNReal>=sn]) - len(SNReal[SNReal>=sn+0.1]))
			else:
				Psimexpected = 0.0
				Psimexpected1 = 0.0
				Psimexpected2 = 0.0

			PSimulationExpectedTotalReal = Psimexpected
			PSimulationExpectedTotalE1Real = Psimexpected - Psimexpected1
			PSimulationExpectedTotalE2Real = Psimexpected2 - Psimexpected
			PPoissonExpectedTotalReal = np.interp(sn,bins,ProbPoissonExpected)
			PPoissonExpectedTotalE1Real = ProbPoissonExpectedE1[index]
			PPoissonExpectedTotalE2Real = ProbPoissonExpectedE2[index]

			NewSource = [
							source[0][np.argmax(source[3])],
							source[1][np.argmax(source[3])],
							source[2][np.argmax(source[3])],
							source[3][np.argmax(source[3])],
							source[4],
							np.interp(source[3][np.argmax(source[3])],bins,y),
							np.interp(source[3][np.argmax(source[3])],bins,ProbNegativeOverPositive),
							np.interp(source[3][np.argmax(source[3])],bins,ProbPoisson),ErrorPSimulation_1,
							ErrorPSimulation_2,
							ErrorPPoisson_1,
							ErrorPPoisson_2,
							PNegativeTotalE1Real,
							PNegativeTotalE2Real,
							PNegativeDifTotalReal,
							PNegativeDifTotalE1Real,
							PNegativeDifTotalE2Real,
							PSimulationExpectedTotalReal,
							PSimulationExpectedTotalE1Real,
							PSimulationExpectedTotalE2Real,
							PPoissonExpectedTotalReal,
							PPoissonExpectedTotalE1Real,
							PPoissonExpectedTotalE2Real
						]
			SourcesTotalPos.append(NewSource)

	for source in Sources_realNeg:
		if max(source[3])>=args.MinSN:
			sn = max(source[3])
		
			aux,ErrorPSimulation_1,ErrorPSimulation_2 = LineSeekerFunctions.GetPoissonErrorGivenMeasurements(np.interp(sn,bins,y)*N_simulations2,N_simulations2)
			if N_simulations2==0:
				ErrorPSimulation_1 = 0.0
				ErrorPSimulation_2 = 0.0

			index = np.argmin(abs(bins - sn))
			ErrorPPoisson_1 = ProbPoissonE1[index]
			ErrorPPoisson_2 = ProbPoissonE2[index]

			PNegativeTotalE1Real = np.interp(sn,bins,ProbNegativeOverPositiveE1)
			PNegativeTotalE2Real = np.interp(sn,bins,ProbNegativeOverPositiveE2)
			PNegativeDifTotalReal = np.interp(sn,bins,ProbNegativeOverPositiveDif)
			PNegativeDifTotalE1Real = np.interp(sn,bins,ProbNegativeOverPositiveDifE1)
			PNegativeDifTotalE2Real = np.interp(sn,bins,ProbNegativeOverPositiveDifE2)

			if N_simulations2>0:
				Rate = np.interp(sn,bins,yExpected) - np.interp(sn+0.1,bins,yExpected)
				RateSigma = 0.5*np.sqrt(np.sum(np.power(np.array([np.interp(sn,bins,yExpectedSigma),np.interp(sn+0.1,bins,yExpectedSigma)]),2)))
				auxSimulationExpected = [Rate - RateSigma,Rate,Rate + RateSigma]
				Psimexpected = 1.0 - max(0,len(SNRealNeg[SNRealNeg>=sn]) - len(SNRealNeg[SNRealNeg>=sn+0.1]) - auxSimulationExpected[1])*1.0/(len(SNRealNeg[SNRealNeg>=sn]) - len(SNRealNeg[SNRealNeg>=sn+0.1]))
				Psimexpected1 = 1.0 - max(0,len(SNRealNeg[SNRealNeg>=sn]) - len(SNRealNeg[SNRealNeg>=sn+0.1]) - auxSimulationExpected[0])*1.0/(len(SNRealNeg[SNRealNeg>=sn]) - len(SNRealNeg[SNRealNeg>=sn+0.1]))
				Psimexpected2 = 1.0 - max(0,len(SNRealNeg[SNRealNeg>=sn]) - len(SNRealNeg[SNRealNeg>=sn+0.1]) - auxSimulationExpected[2])*1.0/(len(SNRealNeg[SNRealNeg>=sn]) - len(SNRealNeg[SNRealNeg>=sn+0.1]))
			else:
				Psimexpected = 0.0
				Psimexpected1 = 0.0
				Psimexpected2 = 0.0

			PSimulationExpectedTotalReal = Psimexpected
			PSimulationExpectedTotalE1Real = Psimexpected - Psimexpected1
			PSimulationExpectedTotalE2Real = Psimexpected2 - Psimexpected
			PPoissonExpectedTotalReal = np.interp(sn,bins,ProbPoissonExpected)
			PPoissonExpectedTotalE1Real = ProbPoissonExpectedE1[index]
			PPoissonExpectedTotalE2Real = ProbPoissonExpectedE2[index]

			NewSource = [
							source[0][np.argmax(source[3])],
							source[1][np.argmax(source[3])],
							source[2][np.argmax(source[3])],
							source[3][np.argmax(source[3])],
							source[4],
							np.interp(source[3][np.argmax(source[3])],bins,y),
							np.interp(source[3][np.argmax(source[3])],bins,ProbNegativeOverPositive),
							np.interp(source[3][np.argmax(source[3])],bins,ProbPoisson),ErrorPSimulation_1,
							ErrorPSimulation_2,
							ErrorPPoisson_1,
							ErrorPPoisson_2,
							PNegativeTotalE1Real,
							PNegativeTotalE2Real,
							PNegativeDifTotalReal,
							PNegativeDifTotalE1Real,
							PNegativeDifTotalE2Real,
							PSimulationExpectedTotalReal,
							PSimulationExpectedTotalE1Real,
							PSimulationExpectedTotalE2Real,
							PPoissonExpectedTotalReal,
							PPoissonExpectedTotalE1Real,
							PPoissonExpectedTotalE2Real
						]
			SourcesTotalNeg.append(NewSource)

SNFinalPos = LineSeekerFunctions.GetFinalSN(SourcesTotalPos,PixelsPerBMAJ)
SNFinalNeg = LineSeekerFunctions.GetFinalSN(SourcesTotalNeg,PixelsPerBMAJ)

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
			aux = LineSeekerFunctions.GetSourcesFromFiles(glob.glob(folder+'/*_pos.*'),args.MinSN,PixelsPerBMAJ)

			aux_sn = []
			for source in aux:
				aux_sn.append(max(source[3]))
			aux_sn = np.array(aux_sn)
			SimulatedSourcesTotal.append(aux_sn)

			aux = LineSeekerFunctions.GetSourcesFromFiles(glob.glob(folder+'/*_neg.*'),args.MinSN,PixelsPerBMAJ)

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

	bins,ProbPoisson,ProbNegativeOverPositive,PurityPoisson,NPositive,Nnegative,Nnegative_e1,Nnegative_e2,NegativeFitted,NnegativeReal,ProbPoissonE1,ProbPoissonE2,ProbNegativeOverPositiveE1,ProbNegativeOverPositiveE2,ProbNegativeOverPositiveDif,ProbNegativeOverPositiveDifE1,ProbNegativeOverPositiveDifE2,ProbPoissonExpected,ProbPoissonExpectedE1,ProbPoissonExpectedE2 = LineSeekerFunctions.GetPoissonEstimates(bins,SNFinalPos,SNFinalNeg,args.LimitN,args.MinSN)
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
	plt.xticks(np.arange(int(args.MinSN),max(bins)-0.1,1))
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

FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN,FinalPSimultionE1,FinalPSimultionE2,FinalPPoissonE1,FinalPPoissonE2,FinalPuritySimulationE1,FinalPuritySimulationE2,FinalpNegDiv,FinalpNegDivE1,FinalpNegDivE2,FinalpSimExp,FinalpSimExpE1,FinalpSimExpE2,FinalpPoiExp,FinalpPoiExpE1,FinalpPoiExpE2 = LineSeekerFunctions.GetFinalCandidates(SourcesTotalPos,PixelsPerBMAJ)

hdulist =   fits.open(args.Cube,memmap=True)
w = wcs.WCS(hdulist[0].header)

[ra,dec,freq,stoke] =  w.all_pix2world(FinalX,FinalY,FinalChannel,np.zeros_like(FinalChannel),0)
c = []
for i in range(len(ra)):
  c.append(SkyCoord(ra[i], dec[i], frame='icrs', unit='deg'))

Output = open('LineCandidatesPositive.dat','w')
Output.write('#ID RA DEC FREQ SN PSim PSimE1 PSimE2 PCumNeg PCumNegE1 PCumNegE2 PPois PPoisE1 PPoisE2 PSimExp PSimExpE1 PSimExpE2 PDifNeg PDifNegE1 PDifNegE2 PPoisExp PPoisExpE1 PPoisExpE2\n')

for i in range(len(FinalX)):
  k = i + 1
  i = len(FinalX)-i-1
  
  Line = args.SurveyName+'-'+args.Wavelength+'mm.'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '
  Line = Line + c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '
  Line = Line +format(FinalPuritySimulation[i],'.2f') +' '+format(FinalPSimultionE1[i],'.2f')+' '+format(FinalPSimultionE2[i],'.2f')+' '
  Line = Line + str(round(FinalPurityNegative[i],2))+' '+ str(round(FinalPuritySimulationE1[i],2))+' '+ str(round(FinalPuritySimulationE2[i],2))+' '
  Line = Line +format(FinalPurityPoisson[i],'.2f')+' '+format(FinalPPoissonE1[i],'.2f')+' '+format(FinalPPoissonE2[i],'.2f')+' '
  Line = Line +format(FinalpSimExp[i],'.2f')+' '+format(FinalpSimExpE1[i],'.2f')+' '+format(FinalpSimExpE2[i],'.2f')+' '
  Line = Line +format(FinalpNegDiv[i],'.2f')+' '+format(FinalpNegDivE1[i],'.2f')+' '+format(FinalpNegDivE2[i],'.2f')+' '
  Line = Line +format(FinalpPoiExp[i],'.2f')+' '+format(FinalpPoiExpE1[i],'.2f')+' '+format(FinalpPoiExpE2[i],'.2f')+'\n'
  Output.write(Line)

Output.close()


######## Negatives ########
FinalX,FinalY,FinalChannel,FinalPuritySimulation,FinalPurityNegative,FinalPurityPoisson,FinalSN,FinalPSimultionE1,FinalPSimultionE2,FinalPPoissonE1,FinalPPoissonE2,FinalPuritySimulationE1,FinalPuritySimulationE2,FinalpNegDiv,FinalpNegDivE1,FinalpNegDivE2,FinalpSimExp,FinalpSimExpE1,FinalpSimExpE2,FinalpPoiExp,FinalpPoiExpE1,FinalpPoiExpE2 = LineSeekerFunctions.GetFinalCandidates(SourcesTotalNeg,PixelsPerBMAJ)


[ra,dec,freq,stoke] =  w.all_pix2world(FinalX,FinalY,FinalChannel,np.zeros_like(FinalChannel),0)
c = []
for i in range(len(ra)):
  c.append(SkyCoord(ra[i], dec[i], frame='icrs', unit='deg'))

Output = open('LineCandidatesNegative.dat','w')
Output.write('#ID RA DEC FREQ SN PSim PSimE1 PSimE2 PCumNeg PCumNegE1 PCumNegE2 PPois PPoisE1 PPoisE2 PSimExp PSimExpE1 PSimExpE2 PDifNeg PDifNegE1 PDifNegE2 PPoisExp PPoisExpE1 PPoisExpE2\n')

for i in range(len(FinalX)):
  k = i + 1
  i = len(FinalX)-i-1
  Line = args.SurveyName+'-'+args.Wavelength+'mm.Neg'+str(k).zfill(2)+' '+c[i].to_string('hmsdms',sep=':',precision=3).split()[0]+' '
  Line = Line + c[i].to_string('hmsdms',sep=':',precision=3).split()[1]+' '+str(round(freq[i]/1e9,3)).zfill(3)+' '+str(round(FinalSN[i],1))+' '
  Line = Line +format(FinalPuritySimulation[i],'.2f') +' '+format(FinalPSimultionE1[i],'.2f')+' '+format(FinalPSimultionE2[i],'.2f')+' '
  Line = Line + str(round(FinalPurityNegative[i],2))+' '+ str(round(FinalPuritySimulationE1[i],2))+' '+ str(round(FinalPuritySimulationE2[i],2))+' '
  Line = Line +format(FinalPurityPoisson[i],'.2f')+' '+format(FinalPPoissonE1[i],'.2f')+' '+format(FinalPPoissonE2[i],'.2f')+' '
  Line = Line +format(FinalpSimExp[i],'.2f')+' '+format(FinalpSimExpE1[i],'.2f')+' '+format(FinalpSimExpE2[i],'.2f')+' '
  Line = Line +format(FinalpNegDiv[i],'.2f')+' '+format(FinalpNegDivE1[i],'.2f')+' '+format(FinalpNegDivE2[i],'.2f')+' '
  Line = Line +format(FinalpPoiExp[i],'.2f')+' '+format(FinalpPoiExpE1[i],'.2f')+' '+format(FinalpPoiExpE2[i],'.2f')+'\n'
  Output.write(Line)

Output.close()

print 'EPS used in DBSCAN:',PixelsPerBMAJ

