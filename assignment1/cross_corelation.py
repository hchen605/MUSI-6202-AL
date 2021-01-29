#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:04:44 2021

@author: hsin-hung.chen
"""
"""
Question 1: Correlation Implementation: [20]

Write a python function z = crossCorr(x, y) where x, y and z are numpy arrays of floats.

Write a python function x = loadSoundFile(filename) that takes a string and outputs a numpy array of floats - if the file is multichannel you should grab just the left channel.

Create a main function that uses these functions to load the following sound files and compute the correlation between them, plotting the result to file results/01-correlation.png

Question 2: Finding snare location: [20]

Using the correlation, write a function pos = findSnarePosition(snareFilename, drumloopFilename) that takes the string filenames for the snare and drumloop and outputs a regular python list of sample positions of the best guess for the snare position in the drumloop

Save the result in a text file results/02-snareLocation.txt
"""

import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


def crossCorr(x, y):
    """
    print('length = (', x.size, y.size,')')
    z = np.correlate(x, y, 'full')              #this not work, TBC
    print('corr size = ', z.size)
    z = z/np.max(z) #normalized
    return z
"""
#reference another approach,
    corr_length = len(x) + len(y) - 1
    corr = np.zeros(corr_length)
    padded_x = np.hstack((np.zeros((len(y)-1)), x, np.zeros(len(y) - 1))) #zero padding before convolution

    for n in range(corr_length):
        corr[n] = np.dot(padded_x[n:n+len(y)], y)

    z = corr/np.max(corr)

    return z



def loadSoundFile(filename):
    _, audio = read(filename)

    if (audio.shape[1] > 1):#2 channel
        return audio[:, 0]
    else:
        return audio

"""
def loadSoundFile(filename):                       ---------> this not work, TBC
    
    f = wave.open(filename,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(params[:4])
    #print(params[:])
    strData = f.readframes(nframes)
    print(f.getnframes)
    waveData = np.fromstring(strData,dtype=np.float)
    print(waveData.size)
    
    waveData = waveData*1.0/(max(abs(waveData)))#
    print(len(waveData))
    #waveData = np.reshape(waveData,[nframes,nchannels])
    waveData.shape = -1, 2
    waveData = waveData.T
    f.close()
    # plot the wave
    time = np.arange(0,)*(1.0 / framerate)
    plt.plot(time,waveData[0])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Single channel wavedata")
    plt.grid('on')#標尺，on：有，off:無。

    return waveData
"""

def findSnarePosition(snareFilename, drumloopFilename):

    snare = loadSoundFile(snareFilename)
    drums = loadSoundFile(drumloopFilename)
    
    pos = crossCorr(drums, snare)
    print('corr array = ',pos)
    pos_snare = np.where(pos > 0.8, pos, 0) # ideal max = 1
    #maxElement = numpy.amax(pos)
    print('pos_snare = ', pos_snare)
    return pos_snare


if __name__ == "__main__":
    
    drums = loadSoundFile("drum_loop.wav")
    snare = loadSoundFile("snare.wav")
    
    pos = findSnarePosition("snare.wav", "drum_loop.wav")
    
    f1 = open('./results/02-snareLocation.txt','w')
    f1.write('#position of snare\n')
    #f1.write('[')
    snareLocation = [] #position list
    for i in range(len(pos)):
        if pos[i] > 0:
            i = i - len(snare)+1
            snareLocation.append(i)
            f1.write(str(i))
            f1.write('\n')
    
    
    
    sample = range(len(drums))
    corr = crossCorr(drums, snare)
    
    plt.plot(sample, corr[len(snare)-1:])  #
    plt.xlabel("Time (In samples)")
    plt.ylabel("Normalized Correlation Coefficient")
    plt.title("Cross Correlation between Drum loop and Snare")
    plt.savefig('./results/01-correlation.png')
    plt.show()
    

#print("hello")
#snare = loadSoundFile('snare.wav')
#drum = loadSoundFile('drum_loop.wav')
#print(x[88198])