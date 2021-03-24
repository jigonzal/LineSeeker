# LineSeeker
Python Codes that search for emission line candidates in ALMA data cubes

Work in progress... updated to Python 3

Please contact Jorge González-López at jorge.gonzalezl@mail.udp.cl for comments an questions.

The code has two methods to estimate the reliability of the emission lines; using the negative data or simulations. 

In the case you only want to use the negative data as reference: 

- First run the the SearchLine.py code to look for emission lines features in the data cube. The output will be stored in a folder and used by the other codes.
- Finally, run GetLineCandidates.py. This code will take the output from the previous step and deliver a list of emission line candidates. It gives probabilities of the line being false using the negative data. 


In the case you also want to use simulations (this can take a lot of time):
- First run the the SearchLine.py code to look for emission lines features in the data cube. The output will be stored in a folder and used by the other codes.
- The second step is to run the SearchLine algorithm over simulated cubes to have an estimate of the significance of the line candidates S/N. Run SimulateCubeSearch.py
- Finally, run GetLineCandidates.py. This code will take the outputs from the two previous steps and deliver a list of emission line candidates. It gives probabilities of the line being false estimated using the negative data and simulations. 

To keep in mind:

- When the cube is well behaved and symmetrical with respect to zero (difficult to say by eye) is ok to use only the negative data as reference. If you have doubts about some of the candidates, then also run the simulations and get the probabilitites again. When using simulations we are assuming that the noise is Gaussian.


Example using BRI1202:

- First run "python SearchLine.py -h" to get the options. 

- The most important parameter is -Cube and is the only one forced to run the script. 

- Run "python SearchLine.py -Cube BR1202.line.fits" and the script will run with the default parameters. 
Output:
```
(base) jgonzal@MacBook-Pro-de-Jorge LineSeeker % python SearchLine.py -Cube BR1202.line.fits
#################### Checking inputs.... ####################
*** Cube BR1202.line.fits found ***
*** Creating Directory OutputLineSearch  ***
*** The value for MaxSigmas of 10 is ok ***
*** The value for MinSN of 5.0 is ok ***
*** Will not use Continuum image to create mask ***
*** Using Gaussian Kernel ***
*** MaxSigmas should be of the order of  31 to detect a line width FWHM ~ 1000 km/s ***
*** A rough guesstimate to use as MinSN is 2.9 ***
####################################################################################################
```
- By default the code will suggest some parameter to search for lines with FWHM approx 1000 km/s and an estimate of the MinSN to use. 
- SearchLine.py creates a default output folder and it will not rewrite it. This is just a safe measure to not delete data by accident. There is an option to change the name of the output folder.
- Run "rm -rf OutputLineSearch"
- Run "python SearchLine.py -Cube BR1202.line.fits -MinSN 2.9 -MaxSigmas 31"

- Now run "python GetLineCandidates.py -h" to get the options:
- Now run "python GetLineCandidates.py -Cube BR1202.line.fits -MinSN 2.9 -MaxSigmas 31"
Here I am using the same parameters as for SearchLine.py but can be changed. 

The output in the terminal for GetLineCandidates.py is very important, some warning from python can and will appear. 

What is important is this:
```
--------------------------------------------------
for sigma 1
S/N NDetected Fraction Nsimulations ExpectedNumberPerCube Error
Min SN to do the fit: 2.9 , Number of usable bins: 14
Min SN to do the fit: 3.0 , Number of usable bins: 13
Min SN to do the fit: 3.1 , Number of usable bins: 12
Min SN to do the fit: 3.2 , Number of usable bins: 11
Min SN to do the fit: 3.3 , Number of usable bins: 10
Min SN to do the fit: 3.4 , Number of usable bins: 9
Min SN to do the fit: 3.5 , Number of usable bins: 8
Min SN to do the fit: 3.6 , Number of usable bins: 7
Min SN to do the fit: 3.7 , Number of usable bins: 6
--------------------------------------------------
```
This will appear for each of the convolutions used in the search, 31 in this case. Here I'm showing the one for n=1. The code will try to use a SN cut so that it has at least six bins with more than 20 (default number) negative line candidates. I do this to have a good reference for the negative data without suffering from incomplete sampling of low SN negative lines. 

- The most important parameter to pay attention is MinSN. If we use a value to high then we will not have the six bins with more than 20 negative line candidates. If we use a value too low, then the clustering algorithm will start blending the lines. If not enough bins, the script will print a message. I think it is safe to have at least 3-4 bins for the fit of the negative counts. If the bins go below that, the script tries to force the fitting. 
```
--------------------------------------------------
for sigma 23
S/N NDetected Fraction Nsimulations ExpectedNumberPerCube Error
Min SN to do the fit: 2.9 , Number of usable bins: 5
*** We are using  5  points for the fitting of the negative counts ***
*** We usually get good results with 6 points, try reducing the parameter -MinSN ***
--------------------------------------------------
```
- In this example, the last iteration returns:
```
--------------------------------------------------
for sigma 30
S/N NDetected Fraction Nsimulations ExpectedNumberPerCube Error
Min SN to do the fit: 2.9 , Number of usable bins: 4
--------------------------------------------------
```
In this case I would repeat the search using MinSN=2.7 in order to add a couple of bins for the final search. 

- When runnign with MinSN=2.7, pay attention to the generated plots, specially to NumberPositiveNegative_0.pdf, NumberPositiveNegative_7.pdf and NumberPositiveNegative_29.pdf
These are cumulative histograms with the negative lines (orange points), positive lines (green points) and the fit to the negative data (blue line). 

NumberPositiveNegative_0.pdf shows how in the low end of SN, the histogram can start departing from the assumed function. This is produced because some low SN lines can be missed. 
NumberPositiveNegative_7.pdf shows how the data can miss behave. Here we see an "excess" of positive and negative lines with SN>4 with respect to the fitted function. This will produce an overestimate of the significance of some lines. In this case I would also run the simulations just to be sure the lines are reliable. 
NumberPositiveNegative_29.pdf shows how when convolving several channels, the data is better behave and the fitted function better follows the negative lines. 

If you see that the plots fail to show the points or the orange points are too different from the fitted function, then try and changing some parameters such as:
```
-MinSN MINSN          Minimum S/N value to save in the outputs. A good value depends on each data cube, reasonable values are bettween 3.5 and 6 [Default:5.0]
-LimitN LIMITN        Limit for the number of detection above certain S/N to be used in the fitting of the negative counts [Default:20]

-UserEPS {True,False}  Whether to use EPS value entered from user otherwise use number of pixels per bmaj [Default:False]
-EPS EPS              EPS value to use if User sets -UserEPS to True [Default:5.0]
```







