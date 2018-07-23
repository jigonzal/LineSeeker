# LineSeeker
Python Code that search for emission line candidates in ALMA data cubes

Work in progress...

Please contact Jorge González at jorge.gonzalezl@mail.udp.cl for comments an questions.

The workflow goes as follow: 

- First run the the SearchLine.py code to look for emission lines features in the cube. The output will be stored in a folder and used by the next codes.
- The second step is to run the SearchLine algorithm over 50 simulated cubes to have an estimate of the significance of the line candidates S/N. Run SimulateCubeSearch.py
- Finally, run GetLineCandidates.py. This code will take the outputs from the two previous steps and deliver a list of emission line candidates. It gives probabilities of the line being false estimated using the negative data and simulations. 


TODO:
- As of now the clustering algorithm is tuned to work with pixel ~1/5 of the beam size and channel width of around 1/3 to 1/5 of the minimum line width. In the future I will generalize the parameters for DBSCAN (eps=10, min_samples=1,leaf_size=30) or leave them as required parameters.

Update: I have modified the code so the eps parameter is equal to bmaj/PixelSize. If MaxSigmas=1, then eps=1 (for continuum searches)
