#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri FEB 5 15:04:44 2021

@author: hsin-hung.chen
"""
"""
Question 1: Time Domain Convolution: [30]

In this part, you will write a method to compute the sample by sample time domain convolution of 2 signals. The basic formula for discrete convolution can be found here: http://en.wikipedia.org/wiki/Convolution

Write a python function y = myTimeConv(x,h) that computes the sample by sample time domain convolution of two signals. 'x' and 'h' are the signal and impulse response respectively and must be NumPy arrays. 
'y' is the convolution output and also must be a NumPy array (single channel signals only).  [15]
If the length of 'x' is 200 and the length of 'h' is 100, what is the length of 'y' ? It is sufficient to only provide the answer in a comment above the convolution implementation. [5]
In your main script define 'x' as a DC signal of length 200 (constant amplitude of 1) and 'h' as a symmetric triangular signal of length 51 (0 at the first and last sample and 1 in the middle). 
Add a function call to myTimeConv() in your script to compute 'y_time' as the time-domain convolution of 'x' and 'h' as defined above. Plot the result (label the axes appropriately) and save in the results folder [10]

"""

import numpy as np
#import wave
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy import signal
import time

"""
If the length of 'x' is 200 and the length of 'h' is 100, what is the length of 'y' ? 299
"""
def myTimeConv(x,h):
    
    conv_length = len(x) + len(h) - 1
    h_i = h[::-1] #invese array
    y = np.zeros(conv_length)
    padded_x = np.hstack((np.zeros((len(h)-1)), x, np.zeros(len(h) - 1))) #zero padding before convolution
    
    
    for n in range(conv_length):
        y[n] = np.dot(padded_x[n:n+len(h)], h_i)
    
    return y


def triangle(length, amplitude):
     section = int((length -1)/2)
     y = np.zeros(length, dtype=np.float16) #
     y = y.astype(np.float16)
     for direction in (1, -1):
         if(direction == 1):
            for i in range(section):
                y[i] =  i * (amplitude / section) * direction
         else:
            for i in range(section):
                y[i+section] =  (amplitude - (i * (amplitude / section))) 
     #y[section] = amplitude
            
     return y

def CompareConv(x, h):
    
    import time
    
    start_time_0 = time.time()
    y0 = myTimeConv(x,h)
    end_time_0 = time.time()
    t0 = end_time_0 - start_time_0
    
    start_time_1 = time.time()
    y1 = signal.convolve(x, h, mode='full')
    end_time_1 = time.time()
    t1 = end_time_1 - start_time_1
    
    sample = range(len(y1))
    plt.plot(sample, y1)  #
    plt.xlabel("Time (In samples)")
    plt.ylabel("Amplitude")
    plt.title("Convolution")
    plt.savefig('./results/02_scipy_convolution.png')
    
    diff = y0 - y1
    #print(diff)
    m = np.mean(diff)
    mabs = np.mean(np.absolute(diff))
    stdev = np.std(diff, ddof=1)
    time = (t0, t1)
    
    return m, mabs, stdev, time

def loadSoundFile(filename):
    _, audio = read(filename)

    #if (audio.shape[1] > 1):#2 channel
        #return audio[:, 0]
    #else:
    return audio
    
    
    
x = np.ones(200)
h = triangle(51,1)
 
y_time = myTimeConv(x,h)

(m, mabs, stdev, time) = CompareConv(x, h)




sample = range(len(y_time))
plt.plot(sample, y_time)  #
plt.xlabel("Time (In samples)")
plt.ylabel("Amplitude")
plt.title("Convolution")
plt.savefig('./results/01_convolution.png')
plt.show()


## Q2


piano = loadSoundFile("piano.wav")
impulse = loadSoundFile("impulse-response.wav")


"""
plt.plot(range(len(piano)), piano)  #
plt.xlabel("Time (In samples)")
plt.ylabel("Amplitude")
plt.title("piano")
plt.savefig('./results/piano.png')
plt.show()

plt.plot(range(len(impulse)), impulse)  #
plt.xlabel("Time (In samples)")
plt.ylabel("Amplitude")
plt.title("impulse")
plt.savefig('./results/impulse.png')
plt.show()
"""



(m, mabs, stdev, time) = CompareConv(piano, impulse)



f = open('./results/CompareConv.txt','w')
f.write('#compare convolution between handcode funcation and scipy function\n')
f.write('m = ') 
f.write(str(m)) 
f.write('\n')
f.write('mabs = ')
f.write(str(mabs))
f.write('\n')
f.write('stdev = ')
f.write(str(stdev)) 
f.write('\n')
f.write('time = ')
f.write(str(time)) 
f.write('\n')

f.close()






