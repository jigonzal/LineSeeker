# LineSeeker
Python Codes that search for emission line candidates in ALMA data cubes

Work in progress...

Please contact Jorge González at jorge.gonzalezl@mail.udp.cl for comments an questions.

The workflow goes as follow: 

- First run the the SearchLine.py code to look for emission lines features in the data cube. The output will be stored in a folder and used by the other codes.
- The second step is to run the SearchLine algorithm over simulated cubes to have an estimate of the significance of the line candidates S/N. Run SimulateCubeSearch.py
- Finally, run GetLineCandidates.py. This code will take the outputs from the two previous steps and deliver a list of emission line candidates. It gives probabilities of the line being false estimated using the negative data and simulations. 


