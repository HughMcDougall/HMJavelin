from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from random import random, choice, gauss
from math import sin, cos, log, exp, pi
import os
import chainconsumer
from Data_sim_Utils import correl_time_mult, correl_time, disc_correl

from Data_sim_Utils import read_dir

'''
Data_Deg_Chain_Analysis.py

Takes the chains from Data_Deg_Test.py and generates histograms

HM 28/8/22
Changes:
-31 / 8 
.Now read dirfile to get targets for chain generation instead of hardcoding names
-1/9
.Added ultra high quality and run-test
.
.Fixed bugs in header file write
-8/9
.Changed directory reading to external function

DEPRECATED 11/9. USE chain_analysis.py FROM HERE ON OUT
'''


#MISSINGNO - Set taumax to read
taumax = 250*3
targfolder = "./Data/fakedata/Sim_Batch_Mirror"
targappend="-runtest"

#===============================
#Read Directory for targets
url = targfolder+'/_dir.txt'
nosims, nogrades, targdirs  = read_dir(url)
#===============================

print("Startin'")
fig, axs = plt.subplots(nosims,nogrades)
fig.figsize=(10,5)
for i in range(nosims):
    print("Plots for sim %.2i" %i)
    for quality in range(nogrades):
        #Get location of faked data
        url_folder = targdirs[i][quality]
        chain_url   = url_folder + "chain%s.dat" %targappend

        print("Doing Plots for %s" %url_folder.replace(targfolder,""))

        if not os.path.exists(chain_url):
            print("couldnt find file %s" %chain_url)
            continue #Safety check to avoid loading bad chain

        #Load Chain and Extract Taus
        CHAIN=np.loadtxt(chain_url)
        TAU_1 = CHAIN[:, 2]
        TAU_2 = CHAIN[:, 5]

        TAU_1 = TAU_1[np.where(TAU_1<taumax)]
        TAU_2 = TAU_2[np.where(TAU_2<taumax)]

        #Plot True Values
        ax=axs[i,quality-1]
        ax.hist(TAU_1,histtype="step",density=True,bins=64)
        ax.hist(TAU_2,histtype="step",density=True,bins=64)

        #MISSINGNO - Set these to load ture values from the header.txt
        ax.axvline(250, c='blue')
        ax.axvline(100, c='orange')

        #plot 180 day lines
        for seasonline in range(0,taumax,180):
            ax.axvline(seasonline, c='k', linestyle="--", alpha=0.5)

        ax.set_xlim([0,taumax])
        if i!=range(nosims)[-1]:
            ax.set_xticks([])
        else:
            ax.set_xticks(range(0,taumax,180))
        ax.set_yticks([])

        cfig= chainconsumer.ChainConsumer().add_chain(np.delete(CHAIN,[3,6],1),  parameters=["logtau","sig","Delay1","sigma1","Delay2","sigma2"]).plotter.plot()
        plt.close(cfig)
        cfig.savefig(url_folder+"/contours_all.png", format='png')

        cfig= chainconsumer.ChainConsumer().add_chain(np.vstack([CHAIN[:,2],CHAIN[:,5]]).T,  parameters=["Delay1","Delay2"]).plotter.plot()
        cfig.savefig(url_folder + "/contours_delaysonly.png", format='png')
        plt.close(cfig)

        np.savetxt(X= correl_time_mult(CHAIN), fname = url_folder + "correltimes%s.dat" %targappend)
        np.savetxt(X=np.array([CHAIN.shape[0],len(np.unique(CHAIN, axis=0)),len(np.unique(CHAIN, axis=0))/CHAIN.shape[0]]), fname=url_folder + "acceptance%s.dat" %targappend)


fig.tight_layout()
fig.savefig(targfolder+"/Histograms%s.png" %targappend, format='png')
plt.close(fig)


print("done")

plt.show()