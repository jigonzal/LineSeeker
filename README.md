# LineSeeker
Python Code that search for emission line candidates in ALMA data cubes

Work in progress...

Please contact Jorge González at jorge.gonzalezl@mail.udp.cl for comments an questions.

The workflow goes as follow: 

- First run the the SearchLine.py code to look for emission lines features in the cube
- The second step is to run the SearchLine algorithm over 50 simulated cubes to have an estimate of the significance of the line candidates S/N
- Finally, run GetCandidates.py. This code will take the outputs from the two previous steps and deliver a list of emission line candidates.


TODO:
- As of now the clustering algorithm is tuned to work with pixel ~1/5 of the beam size and channel width of around 1/3 to 1/5 of the minimum line width. In the future I will generalize the parameters for DBSCAN (eps=10, min_samples=1,leaf_size=30) or leave them as required parameters.
