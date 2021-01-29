# MUSI-6202-AL
Digital Signal Processing

Assignment 1
- run directly in the assignment1 folder
- command: python cross_corelation.py

Question 1: Correlation Implementation: [20]

Write a python function z = crossCorr(x, y) where x, y and z are numpy arrays of floats.

Write a python function x = loadSoundFile(filename) that takes a string and outputs a numpy array of floats - if the file is multichannel you should grab just the left channel.

Create a main function that uses these functions to load the following sound files and compute the correlation between them, plotting the result to file results/01-correlation.png

Question 2: Finding snare location: [20]

Using the correlation, write a function pos = findSnarePosition(snareFilename, drumloopFilename) that takes the string filenames for the snare and drumloop and outputs a regular python list of sample positions of the best guess for the snare position in the drumloop

Save the result in a text file results/02-snareLocation.txt
