from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from random import random, choice, gauss
from math import sin, cos, log, exp, pi
import os as os

'''
Data_sim_Utils.py

Generates DRWs and fake measurements to test javelin on

24/8/2022

Changes
31/8 - Added mirror_sample(), changed formatting, moved functions to own file
8/9 - Added some more useful tools
9/9 - Added 'baseline' argument to mirror_sample. Previously used baseline = max(T_mirror)-min(T_mirror), which caused warping
11/9 - Added verbose tag
26/9 - Added 'delay from center' to tophat convolution. Added documenting to DRW_Sim
'''

#===============================
#AGN LIGHTCURVE GENERATORS
#===============================

def DRW_sim(dt, tmax, tau, siginf=1, x0=None, E0=0, method='square'):
    '''
    DRW_sim()

    Generates a damped random walk of set timescale and variability but zero mean

    Arguments:
        dt      float   The timestep of the walk
        tmax    float   The time-length of the simulation
        tau     float   The DRW's damping timescale

    Additional Arguments:
        signinf float   The DRWs inherent stochastic variability. Defaults to 1
        x0      float   The starting position of the walk. If not set, will be randomly generated based on siginf
        E0      float   The 1 sigma uncertainty in x0. If non-zero and with x0 set to some value, is used to randomize starting position around x0
        method  str     The method by which the stochastic elements are generated
                        -square:    Evenly distributed on the domain (-1,1)
                        -norm:      Normal gaussian distribution
                        -flip:      randomized (-1,1)

    Returns as two arrays of length int(tmax/dt):
        Tout    [float] Evenly spaced time values
        Xout    [float] The corresponding values of the DRW
    '''

    N_out = int(tmax // dt + 1)

    Xout = np.zeros(N_out)
    T = np.linspace(0, tmax, N_out)

    #Sim constants that are used multiple times
    k = (dt / tau) ** 0.5 * siginf
    a = (1 - dt / tau)

    if x0 == None:
        x = gauss(0, siginf)
    else:
        if E0 != 0:
            x = gauss(x0, E0)
        else:
            x = x0

    if method == 'square':
        k /= 1 / 6 ** 0.5
    elif method == 'flip':
        k /= (1 / 2) ** 0.5
    elif method == 'norm':
        k /= (1 / 2) ** 0.5
    else:
        raise TypeError("Tried to run sim DRW sim with bad random generator")

    #Main sim loop
    for i in range(0, N_out):
        if method == 'square':
            rand = (random() * 2 - 1)
        elif method == 'flip':
            rand = choice([-1, 1])
        elif method == 'norm':
            rand = gauss(0, 1)
        else:
            raise TypeError("Tried to run sim DRW sim with bad random generator")

        Xout[i] = x
        x = x * a + rand * k

    return (T, Xout)

def tophat_convolve(Tin,Xin, tau, siginf=1, method='square',delay=0,amp=1,width=None, delay_from_center=True):
    '''
    DRW_sim()

    Generates a damped random walk of set timescale and variability but zero mean

    Arguments:
        Tin     [float] Time values of the continuum signal to be convolved. Must be evenly spaced.
        Xin     [float] Signal of continuum signal to be convolved
        tau     float   The DRW's damping timescale

    Additional Arguments:
        signinf float   The DRWs inherent stochastic variability. Defaults to 1
        delay   float   The delay between line and continuum. Measured relative to start of tophat if delay_from_center = False, and middle of tophat if =True
        amp     float   The amplitude of the output signal
        width   float   The tophat width. If None, will set equal to timestep of signal
        method  str     The method by which the stochastic elements are generated
                        -square:    Evenly distributed on the domain (-1,1)
                        -norm:      Normal gaussian distribution
                        -flip:      randomized (-1,1)

    Returns as array of length equal to input signal.
        Xout    [float] The corresponding values of the DRW
    '''

    #Correction for delay being from middle of tophat
    if width !=None and delay_from_center==True:
        delay = delay - width / 2
    dt=Tin[1]-Tin[0]

    # Make tophat function to convolve with
    if width==None or width<dt:
        width=dt
    if width+delay<=dt:
        delay=0
        width=dt
        Npad=1
    else:
        Npad = int((delay + width) // dt)


    Ytop = np.array([0.0 if t<delay or t>delay+width else 1.0 for t in Tin[:Npad]])
    #Normalize
    Ytop=Ytop/Npad

    #Pad out data on either side with DRW sim
    PadLeft =DRW_sim(dt,dt*(Npad-1),tau,siginf,x0=Xin[0],method='square')[1]
    PadRight=DRW_sim(dt,dt*(Npad-1),tau,siginf,x0=Xin[-1],method='square')[1]
    PadLeft =PadLeft[::-1]
    Xpadded = np.hstack([PadLeft, Xin, PadRight])


    #Do convolution
    Xout = np.convolve(Xpadded,Ytop,mode='full') * amp / width
    Xout=Xout[Npad:Npad+len(Xin)]


    #Output
    return(Xout)

#===============================
#MEASUREMENT FAKERS
#===============================

def season_sample(T, X, T_season, dt=None, Eav=0.0, Espread=0.0, Emethod='gauss', garble=True, rand_space=True):
    '''
    season(T,X,T_season)

    Takes time series data T,X and simulates seasonal measurements

    Arguments:
        T           [float] The time values of the input data
        X           [float] The signal values of the input data
        T_season    float   The length of the seasons

    Additional Arguments:
        dt          float   The cadence of measurements within the seasons. If not set, will just use the average spacing of the time series data
        Eav         float   The average measurement error
        Espread     float   The variability of the measurement error
        Emethod     str     The method by which the measurement errors are randomized
                                square:   Measurement errors are evenly distributed on the range Eav +/- Espread
                                gauss:    Measurement errors are randomly generated by a gaussian distribution with mean Eav and standard deviation Espread
        garble      bool    If true, the measurements will be randomized within their uncertainty distributions
        rand_space  bool    If true, measurements are sub-sampled randomly instead of with fixed cadence

    Returns as three arrays:
        Tout        [float] The time series of the measurements
        Xout        [float] The measurments themselves
        Eout        [float] The uncertainties in the measurements
    '''
    assert T_season < max(T), "Season length must be less than observation timeframe"

    dt_true = T[1] - T[0] #Timestep of the data we're sub-sampling
    if dt == None:
        dt = dt_true #If no measurement cadence is given, sub-divide entire set

    assert dt > dt_true, "Cannot sub-sample with cadence faster than initial data"

    Nout = len(T) * int(max(T) / T_season)
    Tout = np.zeros([Nout])
    Xout = np.zeros([Nout])

    i = 0
    for t, x in zip(T, X):
        #Acquire "on-season" datapoints
        ses = int(t // T_season)
        if (ses // 2) * 2 == ses:
            Tout[i] = t
            Xout[i] = x
            i += 1  # Count number of points

    #Trim Empty Selections
    Tout = Tout[:i - 1]
    Xout = Xout[:i - 1]

    if rand_space == False:
        sparseness = int(dt / dt_true)

        Tout = Tout[::sparseness]
        Xout = Xout[::sparseness]
    else:
        r=dt_true / dt
        I=np.where(np.random.rand(len(Tout))<r)

        Tout=Tout[I]
        Xout=Xout[I]

    #Apply Error
    Eout = np.zeros([len(Xout)])
    if Emethod == 'square':
        for j in range(len(Eout)):
            Eout[j] = Eav + (random() * 2 - 1) * Espread
    elif Emethod == 'gauss':
        for j in range(len(Eout)):
            Eout[j] = abs(gauss(Eav, Espread))

    #Shift measurements within E bars if need be
    if garble == True:
        if Emethod == 'square':
            for j in range(len(Eout)):
                Xout[j] += Eout[j] * (random() * 2 - 1)
        elif Emethod == 'gauss':
            for j in range(len(Eout)):
                Xout[j] += gauss(0, Eout[j])

    #Make sure no errors are negative
    Eout=np.abs(Eout)

    return (Tout, Xout, Eout)

def mirror_sample(Tsource,  Xsource, Tmirror,   Xmirror,    Emirror, Emethod='gauss', garble=True, baseline=None, offset=None):
    '''
    season(T,X,T_season)

    Takes time series data T,X and simulates seasonal measurements

    Arguments:
        T_source    [float] The time values of the input data
        X_source    [float] The signal values of the input data

        T_mirror    [float] Times of measurements of the signal to mirror
        X_mirror    [float] Signal values of measurements to mirror
        E_mirror    [float] Errors in measurements to mirror

    Additional Arguments:
        Emethod     str     The method by which the measurement errors are randomized
                                square:   Measurement errors are evenly distributed on the range Eav +/- Espread
                                gauss:    Measurement errors are randomly generated by a gaussian distribution with mean Eav and standard deviation Espread
        garble      bool    If true, the measurements will be randomized within their uncertainty distributions


    Returns as three arrays:
        Tout        [float] The time series of the measurements
        Xout        [float] The measurments themselves
        Eout        [float] The uncertainties in the measurements
    '''

    assert len(Tsource) == len(Xsource), "Input measurements must all be same length"
    assert len(Tmirror) == len(Xmirror) and len(Tmirror) == len(Emirror), "Input measurements must all be same length"

    Nout = len(Tmirror)

    Tout = np.zeros([Nout])
    Xout = np.zeros([Nout])
    Eout = np.zeros([Nout])

    #Scale time measurements to signal baseline
    if baseline==None:  baseline    = Tmirror[-1]-Tmirror[0]
    if offset==None:    offset      = Tmirror[0]
    Tout = (Tmirror - offset ) #* baseline / (max(Tmirror) - offset )

    #Get measurements by linear interpolation
    if Tout[0]==Tsource[0]:
        Xout[0] = Xsource[0]
        istart=1
    else:
        istart=0
    for i in range(istart,Nout):
        J=np.where(Tsource>=Tout[i])[0][0:2]
        j1 = J[0]
        j2 = J[1]
        r = (Tout[i]-Tsource[j1]) / (Tsource[j2]-Tsource[j1])
        Xout[i]= Xsource[j1] * (1-r) + Xsource[j2] * r


    #Scale Erorrs to same width as fake signal
    if np.std(Xmirror)==0:
        Eout=Emirror
    else:
        Eout = Emirror * np.std(Xsource) / max(np.std(Xmirror),0)

    #If garbling, scatter measurements within uncertainty bounds
    if garble:
        if Emethod=="square":
            Xout += (1-2*np.random.rand(size=Nout) )* Eout
        elif Emethod=="gauss":
            Xout += np.random.normal(size=Nout) * Eout

    return (Tout, Xout, Eout)

#================================================================
#Quick simulation functions
#================================================================


#================================================================
if __name__=="__main__":
    Tsource=np.linspace(0,1000,1024*4)
    Xsource = np.sin(Tsource*2*np.pi/100)

    Tseason, Xseason,Eseason = season_sample(Tsource, Xsource, T_season=max(Tsource)/6, dt=max(Tsource)/6/8, Eav=.1, Espread=0.01, Emethod='gauss', garble=False, rand_space=False)
    Tseason-=Tseason[0]
    plt.plot(Tsource, Xsource)
    plt.errorbar(Tseason, Xseason, yerr=Eseason, fmt='x')


    Ttest, Xtest = Tsource[np.where(Tsource < max(Tseason))[0]], Xsource[np.where(Tsource < max(Tseason))[0]]
    Tseason += 1000
    Tmirror, Xmirror, Emirror = mirror_sample(Ttest,Xtest,Tmirror=Tseason, Xmirror=Xseason, Emirror=Eseason, garble=False)

    plt.errorbar(Tmirror, Xmirror, yerr=Emirror, fmt='.')

    plt.show()

#==================================



