import numpy as np
import matplotlib.pylab as plt
from random import random, choice, gauss
from math import sin, cos, log, exp, pi
import os as os
from Data_sim_Utils import *
from reftable import *

'''
Datagen.py

Functions for generating fake AGN data more easily. 
Functions to actually simulate the AGN are in Data_sim_Utils.py, but makes use of classes and objects in reftable.py to make usage easier.
HM - 9/9

Changes

11/9 
-simgen_batch now generate the header files with the run information.

15/9
-Additional Commenting
-Added verbose tag
-Removed from __future__ import division

'''


def simgen(sim_targ_folder, sim_realization, sampling_params= subsampling_grade(), makefigs = True, fig_targ_folder= None, verbose = False, prefix=None):
    '''
    Takes a DRW realization and subsamples it, outputting to sim_targ_folder
    Outputs to a folder 'sim_targ_folder/name/[FILES]'

    Inputs
        Var                 Type
        sim_targ_folder     str             Url to save the subsampled lightcurves to
        sim_realization     [array]         A list containing the true lightcurves that are being subsampled, in the order [T, X_cont, X_line_1,X_line_2]

    Additional Arguments
        Var                 Type            Default
        sampling_params     sampling_params subsampling_grade()     How to subsample the lightcurve
        makefigs            bool            True                    If true, generates errorbar plots of the subampling
        fig_targ_folder     str             None                    Override location for saving the output figures. If false, saves sim_targ_folder/figs
        verbose             bool            False                   If true, generates additional progress statements
    '''

    if not os.path.exists(sim_targ_folder): os.makedirs(sim_targ_folder)

    if isinstance(sampling_params, subsampling_grade):
        mode="fixedquals"
        cadence_cont = sampling_params.cadence_cont
        E_cont       = sampling_params.E_cont
        DE_cont      = sampling_params.DE_cont

        cadence_line = sampling_params.cadence_line
        E_line       = sampling_params.E_line
        DE_line      = sampling_params.DE_line

        tseason     =  sampling_params.tseason
    elif isinstance(sampling_params, data_source):
        mode="mirrorquals"
        Tcs,  Xcs,  Ecs     = sampling_params.T_cont, sampling_params.X_cont,  sampling_params.E_cont
        Tls1, Xls1, Els1    = sampling_params.T_line1, sampling_params.X_line1,  sampling_params.E_line1
        Tls2, Xls2, Els2    = sampling_params.T_line2, sampling_params.X_line2,  sampling_params.E_line2

    #MAKE MEASUREMENTS
    Twalk, Xwalk, Xemit1,Xemit2 = sim_realization

    #Fake a seasonal observation of continuum
    if mode=="fixedquals":
        Tcont,Ycont,Econt       =season_sample(Twalk, Xwalk,  tseason, dt=cadence_cont, Eav=E_cont, Espread=DE_cont, Emethod='gauss', garble=True, rand_space=False)
        Tline1,Yline1,Eline1    =season_sample(Twalk, Xemit1, tseason, dt=cadence_line, Eav=E_line, Espread=DE_line, Emethod='gauss', garble=True, rand_space=False)
        Tline2,Yline2,Eline2    =season_sample(Twalk, Xemit2, tseason, dt=cadence_line, Eav=E_line, Espread=DE_line, Emethod='gauss', garble=True, rand_space=False)

    elif mode=="mirrorquals":
        baseline    = max(Tcs)-min(Tcs)
        offset      = min(Tcs)
        Tcont,Ycont,Econt       =   mirror_sample(Tsource=Twalk, Xsource=Xwalk,  Tmirror=Tcs,  Xmirror=Xcs,  Emirror=Ecs,  Emethod='gauss', garble=True,    baseline = baseline, offset=offset)
        Tline1,Yline1,Eline1    =   mirror_sample(Tsource=Twalk, Xsource=Xemit1, Tmirror=Tls1, Xmirror=Xls1, Emirror=Els1, Emethod='gauss', garble=True,    baseline = baseline, offset=offset)
        Tline2,Yline2,Eline2    =   mirror_sample(Tsource=Twalk, Xsource=Xemit2, Tmirror=Tls2, Xmirror=Xls2, Emirror=Els2, Emethod='gauss', garble=True,    baseline = baseline, offset=offset)

    #Save outputs
    np.savetxt( fname=sim_targ_folder+"/cont.dat",    X=np.vstack([Tcont,Ycont,Econt]).T     )
    np.savetxt( fname=sim_targ_folder+"/line1.dat",   X=np.vstack([Tline1,Yline1,Eline1]).T  )
    np.savetxt( fname=sim_targ_folder+"/line2.dat",   X=np.vstack([Tline2,Yline2,Eline2]).T  )

    if makefigs:
        if verbose: print("Saving images to",fig_targ_folder)

        if fig_targ_folder == None:
            print("No figure target folder. Saving curves directly to output")
            fig_targ_folder = sim_targ_folder
        if not os.path.exists(fig_targ_folder): os.makedirs(fig_targ_folder)

        fig,ax = plt.subplots(3,1,sharex=True)

        ax[0].set_title(sampling_params.name)
        ax[0].errorbar(Tcont, Ycont, Econt, fmt='.', c='b')
        ax[0].set_ylim([np.percentile(Ycont,5) - np.median(Econt)*2,np.percentile(Ycont,95) + np.median(Econt)*2])
        ax[0].grid()

        ax[1].errorbar(Tline1, Yline1, Eline1, fmt='.', c='r')
        ax[1].set_ylim([np.percentile(Yline1,5) - np.median(Eline1)*2,np.percentile(Yline1,95) + np.median(Eline1)*2])
        ax[1].grid()

        ax[2].errorbar(Tline2, Yline2, Eline2, fmt='.', c='g')
        ax[2].grid()
        ax[2].set_ylim([np.percentile(Yline1,5) - np.median(Eline2)*2,np.percentile(Yline2,95) + np.median(Eline2)*2])
        ax[2].set_xlabel("Time (Days)")
        plt.tight_layout()

        if prefix == None: prefix = ''
        fig.savefig(fig_targ_folder+"/"+prefix+sampling_params.name+".png", format="png")
        plt.close(fig)

def simgen_batch(sim_params = sim_params(), SAMPLING_PARAMS=[JavelinExample], nosims=1,   targfolder="./sim", makefigs=True, verbose = False):
    '''
    generates *nosims* realizations of system with properties *sampling_params*
    Subsamples each realisation with [sim_params] and saves all results to targ_folder

    Additional Arguments
        Var             Type        Default
        sim_params      sim_params  sim_params()        The DRW / tophat properties stored in a sim_params() object
        SAMPLING_PARMS  []          [JavelinExamples]   A list of subsampling types, must be a list of sampling_params an/or a data_source objects
        nosims          int         1                   Number of realizations to generate
        targolder       str         ./sim               Location to save the sim batch to
        makefigs        bool        True                If true, generates and saves plots of the generated realizations and subsamplings
        verbose         bool        False               If true, outputs additional progress statements
    '''

    #Load parameters of the AGN signal from sim_params
    dt_sim  = sim_params.dt
    tmax    = sim_params.tmax
    tau     = sim_params.cont_tau
    siginf  = sim_params.siginf_cont

    tau1    = sim_params.delay1
    width1  = sim_params.width1
    amp1    = sim_params.amp1

    tau2    = sim_params.delay2
    width2  = sim_params.width2
    amp2    = sim_params.amp2

    #======================================================
    if verbose: print("Running simgen_batch for target",    targfolder)

    if not os.path.exists(targfolder): os.makedirs(targfolder)

    #Create / write to directory file
    dirfile = open(targfolder + "/_dir.txt", 'w')
    dirfile.write("%i\t%i\n" % (nosims, len(SAMPLING_PARAMS)))
    np.save(arr=[sim_params], file = targfolder+"/sim_params")

    #Create / write header file
    headerfile = open(targfolder + "/_header.txt", 'w')
    headerfile.write('Sim batch writing %i sims to %s\n' %(nosims,targfolder))

    headerfile.write("\n=====================:\n")


    headerfile.write("\nSIMULATION PARAMETERS:\n")
    headerfile.write(sim_params.write_desc())

    headerfile.write("\n=====================:\n")

    headerfile.write("\nSUBSAMPLING PARAMETERS:\n")
    for i,subsampling_grade in enumerate(SAMPLING_PARAMS):
        headerfile.write("-----")
        headerfile.write(subsampling_grade.write_desc())

    headerfile.write("\n=====================:\n")

    headerfile.close()
    #======================================================
    if verbose: print("Settup done.")

    for i in range(nosims):
        #Create folder for each simulation realization
        simno = i + 1
        sim_folder = targfolder+"/sim-%.2i" %simno
        if not os.path.exists(sim_folder): os.makedirs(sim_folder)

        if makefigs:
            figfolder = sim_folder + '/figs'
            if not os.path.exists(figfolder): os.makedirs(figfolder)

        #Generate signals using functions in Data_sim_Utils.py
        Twalk, Xwalk = DRW_sim(dt_sim, tmax, tau, siginf=1, method='square')
        Xemit1 = tophat_convolve(Twalk, Xwalk, tau=tau, siginf=1, method='square', delay=tau1, amp=1, width=width1)
        Xemit2 = tophat_convolve(Twalk, Xwalk, tau=tau, siginf=1, method='square', delay=tau2, amp=1, width=width2)
        Xemit1 /= np.std(Xemit1)  # MISSINGNO - Fix this to use correct amplitude
        Xemit2 /= np.std(Xemit2)  # MISSINGNO - Fix this to use correct amplitude

        sim_realization = [Twalk, Xwalk ,Xemit1 ,Xemit2] # Wrap up to feed to simgen()

        if verbose: print("Simulation %i generated" %i)

        if makefigs:
            #Make plot of underlying curves
            plt.figure()
            plt.plot(Twalk, Xwalk, c='b', lw=0.5, label="Continuum, timescale = %i" % tau)
            plt.plot(Twalk, Xemit1, c='r', lw=0.5, label="line 1, delay = %i, width = %i" % (tau1, width1))
            plt.plot(Twalk, Xemit2, c='g', lw=0.5, label="line 2, delay = %i, width = %i" % (tau2, width2))
            plt.xlabel("time (days)")
            plt.ylabel("Signal Strength (Arb)")
            plt.title("Underlying Curves")
            plt.legend(loc='best')
            plt.grid()
            plt.tight_layout()
            plt.savefig(figfolder + "/TrueCurves.png", format="png")
            plt.close()


        #For each subsamling grade...
        for sampling_params,j in zip(SAMPLING_PARAMS,range(len(SAMPLING_PARAMS))):


            #Get subsampling name and make a folder to store results
            if sampling_params.name !=None:
                sample_folder = sim_folder +"/%.2i-%s" %(j,sampling_params.name)
                if verbose: print("Doing generation for sim %i and sub-sampling %s" % (i,sampling_params.name))
            else:
                sample_folder = sim_folder + "/%.2i-Unnamed Sampling" % j
                if verbose: print("Doing generation for sim %i and sub-sampling %i" %(i,j))

            dirfile.write(sample_folder + "\n") #Write directory name to directory file

            #Do simgen
            simgen(sample_folder,
                   sim_realization  = sim_realization,
                   sampling_params  = sampling_params,
                   makefigs         = True,
                   fig_targ_folder  = figfolder,
                   verbose          = verbose,
                   prefix           = "%.2i_" %j)

    dirfile.close() #Close directory


if __name__ == "__main__":
    #Test simgen
    simgen_batch(SAMPLING_PARAMS=[ClearSignal,JavelinExample,OD_1linegood,OD_2linegood,OD_2linebad], nosims = 2, targfolder="./Data/fakedata/Batch_Test",verbose=True)