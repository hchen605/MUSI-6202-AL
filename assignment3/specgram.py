#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri FEB 5 15:04:44 2021

@author: hsin-hung.chen
"""
"""
Write a function (t,x) = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians) generating a sinusoidal according to the parameters. The outputs x and t are the generated signal and the corresponding time in seconds. Both outputs must be NumPy arrays of the same length.
Add a function call to your script to generate a sine wave with the following parameters: amplitude = 1.0, sampling_rate_Hz = 44100, frequency_Hz = 400 Hz, length_secs = 0.5 seconds, phase_radians = pi/2 radians.
Plot the first 5 ms of the sinusoid generated in Part 2. above (label the axes correctly, time axis must be in seconds)


https://www.itread01.com/article/1532154074.html
https://www.gaussianwaves.com/2015/11/interpreting-fft-results-obtaining-magnitude-and-phase-information/
"""

import numpy as np
import matplotlib.pyplot as plt





def generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians) :
    
    ts = 1/sampling_rate_Hz
    t = np.arange(0, length_secs, ts)
    wt = t * np.pi * 2 * frequency_Hz + phase_radians
    x = amplitude * np.sin(wt)
    
    return t, x
    



def generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians):
    
    ts = 1/sampling_rate_Hz
    t = np.arange(0, length_secs, ts)
    x = np.zeros(len(t))
    for k in range(1, 11): #10 sine addition
        _, s = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz*(2*k-1), length_secs, phase_radians)
        s = s/(2*k-1)
        x = x + s
        
    return t, x*4/np.pi



def computeSpectrum(x, sample_rate_Hz):
    
    samples = len(x)
    #print(samples)
    X_array = np.fft.fft(x)
    X_array_half = X_array[range(int(len(X_array)/2))] #take half
    #freq_bin = int(sample_rate_Hz/samples)
    #f = np.arange(0, int(sample_rate_Hz/2), freq_bin)
    f = np.linspace(0,sample_rate_Hz/2,int(samples/2), endpoint=False) #take only half bins
     
    XAbs = np.abs(X_array_half)
    XPhase = np.angle(X_array_half)
    #XPhase = np.arctan(X_array_half)
    XRe = np.real(X_array_half)
    XIm = np.imag(X_array_half)
    #XPhase = np.arctan(XIm/XRe)
   
    return  f, XAbs, XPhase, XRe, XIm





def generateBlocks(x, sample_rate_Hz, block_size, hop_size):
    
    #ts = 1/sample_rate_Hz
    samples = len(x)
    N = int(np.ceil((samples - block_size)/hop_size)) #block num
    #time = ts * samples
    
    #t = np.arange(0, samples, hop_size) * ts
    t = np.zeros(N)
    X = np.zeros((block_size, N))
    
    for i in range(int(N)):
        t[i] = i * hop_size / sample_rate_Hz
        if (i*hop_size)+block_size < samples:
            X[:,i] = x[(i*hop_size) : (i*hop_size+block_size)]
        else:
            X[0:(samples-i*hop_size),i] = x[(i*hop_size):] #leave others 0x
    
    return t, X


def mySpecgram(x,  block_size, hop_size, sampling_rate_Hz, window_type):
    
    if window_type == 'rect':
        win = np.ones(block_size)
    elif window_type == 'hann':
        win = np.hanning(block_size)
        
    t, X = generateBlocks(x, sampling_rate_Hz, block_size, hop_size)
    #samples = len(x)
    #t_sample = np.arange(0, samples, hop_size)
    
    time_vector = t
    freq_vector, _, _, _, _ = computeSpectrum(X[:,0] * win, sampling_rate_Hz)
    N = np.size(X,1)
    magnitude_spectrogram = np.zeros((int(block_size/2), int(N)))
    x_re = np.zeros(len(x))
    
    for i in range(N):
        x_win = X[:,i] * win
        _, magnitude_spectrogram[:,i], _, XRe, XIm = computeSpectrum(x_win, sampling_rate_Hz)
        #magnitude_spectrogram[:,i] = XRe + XIm * 1j
        x_re[i*hop_size:i*hop_size+block_size] += x_win #reconstruct signal for reference
        
    
    #plt.specgram(x_re, Fs = sampling_rate_Hz) 
    #plt.plot(magnitude_spectrogram)
    plt.figure()
    plt.pcolor(time_vector, freq_vector, magnitude_spectrogram, vmin=0, vmax=600)
    #plt.plotSpecgram(time_vector, freq_vector, magnitude_spectrogram)
    plt.title('Square wave spectrogram: %s' % window_type)
    #plt.title('specgram\n', fontsize = 14, fontweight ='bold')
    plt.xlabel("Time (sample)")
    plt.ylabel("Freq (Hz)")
    plt.colorbar()
    #plt.savefig('./results/04_specgram_{}.png' .format(window_type))
    #plt.show()
    
    
    return freq_vector, time_vector, magnitude_spectrogram


#Q1
amplitude = 1.0
sampling_rate_Hz = 44100
frequency_Hz = 400 
length_secs = 0.5 
phase_radians = np.pi/2


t, x_sine = generateSinusoidal(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)

t_5ms = t[0:int(0.005*sampling_rate_Hz)]
x_5ms = x_sine[0:int(0.005*sampling_rate_Hz)]


plt.plot(t_5ms, x_5ms)  #
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.title("Sine wave")
plt.savefig('./results/01_sine.png')
#plt.show()



#Q2
amplitude = 1.0
sampling_rate_Hz = 44100
frequency_Hz = 400 
length_secs = 0.5 
phase_radians = 0


t, x_square = generateSquare(amplitude, sampling_rate_Hz, frequency_Hz, length_secs, phase_radians)

t_5ms = t[0:int(0.005*sampling_rate_Hz)]
x_5ms = x_square[0:int(0.005*sampling_rate_Hz)]

plt.figure()
plt.plot(t_5ms, x_5ms)  #
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.title("Square wave")
plt.savefig('./results/02_square.png')
#plt.show()
    


#Q3

(f_sine, XAbs_sine, XPhase_sine, XRe_sine, XIm_sine) = computeSpectrum(x_sine, sampling_rate_Hz)
(f_square, XAbs_square, XPhase_square, XRe_square, XIm_square) = computeSpectrum(x_square, sampling_rate_Hz)

#print(f_sine)

plt.subplot(211)
plt.plot(f_sine, XAbs_sine)  
plt.xlabel("Freq (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Sine wave")

plt.subplot(212)
plt.plot(f_sine, XPhase_sine)  
plt.xlabel("Freq (Hz)")
plt.ylabel("Phase (radians)")
#plt.title("FFT of Sine wave")

plt.savefig('./results/03_fft_sine.png')

#print(f_square)

plt.subplot(211)
plt.plot(f_square, XAbs_square)  
plt.xlabel("Freq (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Sqaure wave")

plt.subplot(212)
plt.plot(f_square, XPhase_square)  
plt.xlabel("Freq (Hz)")
plt.ylabel("Phase (radians)")
#plt.title("FFT of Square wave")

plt.savefig('./results/03_fft_square.png')


#Q4

block_size = 2048
hop_size = 1024
freq_vector_rect, time_vector_rect, magnitude_spectrogram_rect = mySpecgram(x_square,  block_size, hop_size, sampling_rate_Hz, 'rect')
#plt.savefig('./results/04_specgram_rect.png')
plt.show()
freq_vector_hann, time_vector_hann, magnitude_spectrogram_hann = mySpecgram(x_square,  block_size, hop_size, sampling_rate_Hz, 'hann')
#plt.savefig('./results/04_specgram_hann.png')
plt.show()

#Q5 ???






