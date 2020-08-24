# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:33:17 2020

@author: Shayani
"""
import numpy as np
import pylab as py
from scipy.fftpack import rfft
from scipy.signal import hann,resample
import soundfile as sf
#%% initialization
filename = input("Enter filename: ") 
# if file doesn't end in .wav change it to .wav
if filename[len(filename)-4:] != ".wav":
    filename=filename+'.wav'

scale = input("Enter scale (F#maj,Cmin,Bbmin, etc.): ") 
# the blocksize determines the actual data block size that will be FFT'd.
# if it's too short, the accuracy of the procedure will drop significantly,
# especially if the signal is not periodic within that block interval.
blocksize = input("Enter FFT block size (press enter for default): ")

if blocksize=="":
    block=4096  
else:
    block=eval(blocksize)

#%% the tuning of the scales in Hz
#C3 up to C4
C=130.81
Cs=138.59
D=146.83
Ds=155.56
E=164.81
F=174.61
Fs=185.0
G=196.0
Gs=207.65
A=220.0
As=233.08
B=246.94

if scale=="Fmin" or scale=="G#maj" or scale=="Abmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((C,Cs,Ds,F,G,Gs,As))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="F#min" or scale=="Gbmin" or scale=="Amaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((Cs,D,E,Fs,Gs,A,B))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="Gmin" or scale=="Bbmaj" or scale=="A#maj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((C,D,Ds,F,G,A,As))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="G#min" or scale=="Abmin" or scale=="Bmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((Cs,Ds,E,Fs,Gs,As,B))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="Amin" or scale=="Cmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((C,D,E,F,G,A,B))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="A#min" or scale=="Bbmin" or scale=="C#maj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((C,Cs,Ds,F,Fs,Gs,As))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="Bmin" or scale=="Dmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((Cs,D,E,Fs,G,A,B))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="Cmin" or scale=="D#maj" or scale=="Ebmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((C,D,Ds,F,G,Gs,As))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="C#min" or scale=="Dbmin" or scale=="Emaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((Cs,Ds,E,Fs,Gs,A,B))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="Dmin" or scale=="Fmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((C,D,E,F,G,A,As))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="D#min" or scale=="F#maj" or scale=="Gbmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((Cs,Ds,F,Fs,Gs,As,B))
    scale2=scale1*2.0
    scale3=scale1/2.0
if scale=="Emin" or scale=="Gmaj":
    #first scale starting on for example F3 as the middle octave
    scale1=np.array((C,D,E,Fs,G,A,B))
    scale2=scale1*2.0
    scale3=scale1/2.0
#combine the octaves for one big frequency list reference.    
scale_comb=np.concatenate((scale1,scale2,scale3))

#%%    
#read in the file
data, sr = sf.read(filename)

#get number of channels
try:
    ch=len(data[0,])
except:
    ch=1

#check if mono, if not provides option for conversion.
if ch != 1:
    q=input("Source must be mono. Would you like to sum to mono [y/n]? ")
    if q=="y" or q=="yes" or q =="":
        print(' ')
        print('Summing to mono...')
        L=data[:,0]
        R=data[:,1]
        n=len(data)
        data=np.zeros((n,1))
        data=L/2.0+R/2.0
    else:
        print(' ')
        raise Exception("Source must be mono.")

#number of pieces
pieces=len(data)//block

#total length of data
n=len(data)

#size of each chunk
chunk=n//pieces

#zeros for zero padding
nz=100000

#array for the data blocks
data_cut=np.zeros((pieces,n//pieces))

#array for adjusted data which is resampled
data_adj=[None]*pieces

#array for x frequencies
xfreq_adj=[None]*pieces

#Arrays for FFT cuts
fft_cut=np.zeros((pieces,nz+n//pieces))

#adjusted FFT list
fft_adj=[None]*pieces
#xfreq array for frequency look up
xfreq=np.linspace(0,sr//2,n//pieces+nz)

#cut the data into pieces
for i in range (pieces):
    data_cut[i][0:n//pieces]=data[i*(n//pieces):(i+1)*(n//pieces)]

#find the fundamental frequency. 
def peakfind(x,y):
    return y[np.argmax(x)]

#array containing the resampling factors
factor=np.zeros(pieces)    

#Here, we a quick FFT and peak findto check how far (in %) the fundamental 
#of the input is from the nearest frequency in the scale, and then we use scipy 
#resample to snap the input freq to the scale freq, but only if its within 20% 
#of the nearest freq, because we want to avoid significant changes in the speed 
#to preserve the same rhythm more or less. 
print('Tuning your input to the key of ' + scale + '...')    
for i in range (pieces):
    fft_cut[i]=rfft(data_cut[i],chunk+nz)
    xfreq_adj[i]=np.linspace(0,sr//2,len(fft_cut[i]))
    factor[i]=scale_comb[np.argmin(abs(peakfind(x=abs(fft_cut[i]),\
                    y=xfreq_adj[i])-scale_comb))]/(peakfind(abs(fft_cut[i]),\
                                                        y=xfreq_adj[i]))
    if factor[i]<0.8:
        factor[i]=1.0
    if factor[i]>1.2:
        factor[i]=1.0
    data_adj[i]=resample(data_cut[i],int(chunk/factor[i]))
           
#connect all the pieces
dataP=np.concatenate((data_adj))
#%% write to file
print('Exporting...')
sf.write(filename[:len(filename)-4]+'_tuned.wav',dataP,sr,'PCM_16')
print('Complete!')



