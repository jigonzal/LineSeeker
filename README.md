# LineSeeker
Python Code that search for emission line candidates in ALMA data cubes
Work in progress. Please contact Jorge González at jorge.gonzalezl@mail.udp.cl for comments.

The workflow goes as follow: 

- First run the the SearchLineXX.py code to look for emission lines features in the cube
- The second step is to run the SearchLine algorithm over 50 simulated cubes to have an estimate of the significance of the line candidates
- Run GetCandidatesXX.py. This code will take the outputs from the two previous codes and delivered a list of emission line candidates with.
