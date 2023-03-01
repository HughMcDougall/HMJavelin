from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from random import random, choice, gauss
from math import sin, cos, log, exp, pi
import os as os
from Data_sim_Utils import *
from reftable import unpack_source

'''
Data_sim.py

Generates DRWs and fake measurements to test javelin on

24/8/2022

Changes
-31/8 
.Removed data generations functions to Data_sim_Utils.py
.Created more read/write functions for ease of use
.Writing a batch of sims now creates a .txt header file that contains the sim parameters

DEPRECATED 9/9. Fake sims generated now in Datagen.py
'''


# ===================
#PROPERTIES OF UNDERLYING SIGNALS
# Continuum
tau     = 400  #DRW Timescale
tmax    = 2160
dt_sim  = 0.1 #Sim resolution
tseason = 180 #Only used if mode = "fixedquals"

# Response
tau1    = 250
amp1    = 1
width1  = 30

tau2    = 100
amp2    = 1
width2  = 30

#Simulation Batch Properties
No_Cases = 5
makefigs = True
run_name = "Sim_Batch_Mirror"
targ_folder = "./data/fakedata/" + run_name + "/"

header_desc = "Sim properties matching javelin example. Data at 5 grades."
mode = "mirrorquals"

# ===================
#Measure Quality. User must input files here.
if mode=="mirrorquals":
    datafolder="Data/RealData/"
    contsources  =  ["00 ClearSignal/cont.dat",     "01 Javelin Example/continuum.dat","03 OzDes 1 Line/2925344542_gBand.dat",     "04 OzDes 2 Line Good/2940510474_gBand.txt",    "05 OzDes 2 Line Bad/gband.dat"]
    line1sources =  ["00 ClearSignal/line1.dat",     "01 Javelin Example/yelm.dat",     "03 OzDes 1 Line/2925344542_Hbeta_exp.dat", "04 OzDes 2 Line Good/2940510474_CIV_exp.txt",  "05 OzDes 2 Line Bad/CIV.dat"]
    line2sources =  ["00 ClearSignal/line2.dat",     "01 Javelin Example/zing.dat",     "03 OzDes 1 Line/2925344542_Hbeta_exp.dat", "04 OzDes 2 Line Good/2940510474_MgII.txt",     "05 OzDes 2 Line Bad/MGII.dat"]
    for i in range(len(contsources)):
        contsources[i]  =   datafolder+contsources[i]
        line1sources[i] =   datafolder+line1sources[i]
        line2sources[i] =   datafolder+line2sources[i]

elif mode == "fixedquals":
    QUALS = []
    for quality in range(1,6):
        if quality == 1:  # Javelin Example
            # Continuum
            cadence_cont = 8
            E_cont = 4.3 / 100
            DE_cont = 1.0 / 100

            # Line Measurements
            cadence_line = 8
            E_line = 4.3 / 100
            DE_line = 1.0 / 100
            QUALS.append([cadence_cont,E_cont,DE_cont,cadence_line,E_line,DE_line])
        elif quality == 2:  # Intermediate 1
            # Continuum
            cadence_cont = 8.5
            E_cont = 4.5 / 100
            DE_cont = 2.5 / 100

            # Line Measurements
            cadence_line = 30
            E_line = 20.5 / 100
            DE_line = 5.9 / 100
            QUALS.append([cadence_cont,E_cont,DE_cont,cadence_line,E_line,DE_line])
        elif quality == 3:  # Ozdes 1 line example
            # Continuum
            cadence_cont = 8.5
            E_cont = 10 / 100
            DE_cont = 7 / 100

            # Line Measurements
            cadence_line = 60
            E_line = 44 / 100
            DE_line = 13 / 100
            QUALS.append([cadence_cont,E_cont,DE_cont,cadence_line,E_line,DE_line])
        elif quality == 4:  # OzDes 2 line example, good
            # Continuum
            cadence_cont = 8
            E_cont = 13 / 100
            DE_cont = 8 / 100

            # Line Measurements
            cadence_line = 60
            E_line = 24.0 / 100
            DE_line = 12.0 / 100
            QUALS.append([cadence_cont,E_cont,DE_cont,cadence_line,E_line,DE_line])
        elif quality == 5:  # Ozdes 2 line example, bad
            # Continuum
            cadence_cont = 8.5
            E_cont = 15 / 100
            DE_cont = 5 / 100

            # Line Measurements
            cadence_line = 60
            E_line = 500 / 100
            DE_line = 100 / 100
            QUALS.append([cadence_cont,E_cont,DE_cont,cadence_line,E_line,DE_line])

# ===================
#RUNTIME
if not os.path.exists(targ_folder): os.makedirs(targ_folder)
dirfile = open(targ_folder + "_dir.txt", 'w')

#Write header file
with open(targ_folder+'header.txt', 'w') as f:
    f.write(run_name)
    f.write("\n")
    f.write(header_desc + "\n \n")
    f.write("SIM PROPERTIES \n")
    f.write("============== \n")

    f.write("tau = %f  \n" %tau )
    f.write("baseline = %f \n" %tmax)
    f.write("sim dt = %f \n" %dt_sim)

    f.write("Line 1: \n")
    f.write("line 1 delay = %f \n" %tau1)
    f.write("line 1 width = %f \n" %width1)
    f.write("line 1 amplitude = %f \n" %amp1)

    f.write("Line 2: \n")
    f.write("line 2 delay = %f \n" %tau2)
    f.write("line 2 width = %f \n" %width2)
    f.write("line 2 amplitude = %f \n" %amp2)

    f.write("DATA PROPERTIES \n")
    f.write("============== \n")
    f.write("Measurement generation mode = %s \n" %mode)

    if mode=="mirrorquals":
        f.write("Data Sources: \n")
        for cont,line1,line2 in zip(contsources,line1sources,line2sources):
            f.write("%s \t %s \t %s \n" %(cont,line1,line2))
    elif mode=="fixedquals":
        f.write("Data Degradation: \n")
        f.write("cadence_cont   E_cont  DE_cont cadence_line    E_line  DE_line \n")
        for qual in QUALS:
            f.write("%.5f \t %.5f \t %.5f \t %.5f \t %.5f \t %.5f" %(qual[0], qual[1] , qual[2] , qual[3] , qual[4] , qual[5]))

    f.close()


if mode =="mirrorquals":
    no_quals = len(contsources)
elif mode=="fixedquals":
    no_quals = len(QUALS)
dirfile.write("%i\t%i\n" %(No_Cases,no_quals))

for sim_no in range(No_Cases):

    sim_targ_folder = targ_folder + "sim-%.2i/" % (sim_no)
    if not os.path.exists(sim_targ_folder): os.makedirs(sim_targ_folder)

    #MAKE MEASUREMENTS
    #Generate a DRW
    Twalk,Xwalk = DRW_sim(dt_sim, tmax, tau, siginf=1, method='square')
    #Convolve with tophat
    Xemit1=tophat_convolve(Twalk,Xwalk, tau=tau, siginf=1,  method='square',delay=tau1,amp=1,width=width1)
    Xemit2=tophat_convolve(Twalk,Xwalk, tau=tau, siginf=1,  method='square',delay=tau2,amp=1,width=width2)
    Xemit1 /= np.std(Xemit1)
    Xemit2 /= np.std(Xemit2)
    print("Fake Signals Generated for instance %i" %sim_no)

    if makefigs:
        fig_targ_folder = sim_targ_folder + "figs/"
        if not os.path.exists(fig_targ_folder): os.makedirs(fig_targ_folder)
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
        plt.savefig(fig_targ_folder + "TrueCurves.png", format="png")
        plt.close()

    #RUN OVER DIFFERENT MEASUREMENT GRADES
    for quality in range(1,no_quals+1):

        meas_targ_folder = sim_targ_folder + "quality-%.2i/" %(quality)
        if not os.path.exists(meas_targ_folder): os.makedirs(meas_targ_folder)

        #Fake a seasonal observation of continuum
        if mode=="fixedquals":
            cadence_cont, E_cont, DE_cont, cadence_line, E_line, DE_line = QUALS[quality-1]
            Tcont,Ycont,Econt       =season_sample(Twalk, Xwalk,  tseason, dt=cadence_cont, Eav=E_cont, Espread=DE_cont, Emethod='square', garble=True, rand_space=False)
            Tline1,Yline1,Eline1    =season_sample(Twalk, Xemit1, tseason, dt=cadence_line, Eav=E_line, Espread=DE_line, Emethod='square', garble=True, rand_space=False)
            Tline2,Yline2,Eline2    =season_sample(Twalk, Xemit1, tseason, dt=cadence_line, Eav=E_line, Espread=DE_line, Emethod='square', garble=True, rand_space=False)
        elif mode=="mirrorquals":
            Tcs, Xcs, Ecs = unpack_source(contsources[quality-1])
            Tls1, Xls1, Els1 = unpack_source(line1sources[quality - 1])
            Tls2, Xls2, Els2 = unpack_source(line2sources[quality - 1])

            Tcont,Ycont,Econt       =   mirror_sample(Tsource=Twalk, Xsource=Xwalk, Tmirror=Tcs, Xmirror=Xcs, Emirror=Ecs, Emethod='gauss', garble=True)
            Tline1,Yline1,Eline1    =   mirror_sample(Tsource=Twalk, Xsource=Xwalk, Tmirror=Tls1, Xmirror=Xls1, Emirror=Els1, Emethod='gauss', garble=True)
            Tline2,Yline2,Eline2    =   mirror_sample(Tsource=Twalk, Xsource=Xwalk, Tmirror=Tls2, Xmirror=Xls2, Emirror=Els2, Emethod='gauss', garble=True)

        print("Subsampling And Saving Done for degredation grade %i" %quality)

        np.savetxt( fname=meas_targ_folder+"cont.dat",    X=np.vstack([Tcont,Ycont,Econt]).T     )
        np.savetxt( fname=meas_targ_folder+"line1.dat",   X=np.vstack([Tline1,Yline1,Eline1]).T  )
        np.savetxt( fname=meas_targ_folder+"line2.dat",   X=np.vstack([Tline2,Yline2,Eline2]).T  )

        dirfile.write(meas_targ_folder + "\n")
        print("Fake data output to %s" % meas_targ_folder)

        #Plotting

        if makefigs:
            plt.figure()
            plt.subplot(311)
            plt.title("Sampling Degredation lvl %.2i" %quality)
            plt.errorbar(Tcont, Ycont, Econt, fmt='.', c='b')
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.grid()
            plt.subplot(312)
            plt.errorbar(Tline1, Yline1, Eline1, fmt='.', c='r')
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.grid()
            plt.subplot(313)
            plt.errorbar(Tline2, Yline2, Eline2, fmt='.', c='g')
            plt.grid()
            plt.xlabel("Time (Days)")
            plt.tight_layout()

            plt.savefig(fig_targ_folder+"%.2i.png" %quality, format="png")
            plt.close()

        print("\n")
dirfile.close()
