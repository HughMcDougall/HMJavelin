import numpy as np
import matplotlib.pylab as plt
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model, Rmap_Model
from Data_sim_Utils import *
from reftable import read_dir

'''
Data_Deg_Test.py

Takes data from Data_sim.py and runs through javelin to test for lag recovery strength

HM 28/8/22
Changes:
-31 / 8 
.Now read dirfile to get targets for chain generation instead of hardcoding names
.Now writes a quick .txt file containing the MCM params
.Adjustable variables for continuum search


TO BE SUPERSEDED. UPDATE WITH FUNCTION / OBJECT BASED VERSION - 10/9
'''

#================================
#Params
targfolder = "./Data/fakedata/Sim_Batch_Mirror"
MCMC_quality = 'highqual'
fixed_widths = True #If true, fixed tophat widths a p_width in MCMC chain
p_width = 30
laglimits = [0, 250 * 3]

#MCMC Params
if MCMC_quality == 'ultrahighqual':
    print("RUNNING WITH ULTRA HIGH QUALITY MCMC PARAMS \n")
    targappend = '-ultrahighqual'
    nwalkers= 1000
    nchain  = 700
    nburn   = 1

    contwalkers = 100
    contchain   = 300
    contburn    = 100
elif MCMC_quality=="highqual":
    print("RUNNING WITH HIGH QUALITY MCMC PARAMS \n")
    targappend = '-highqual'
    nwalkers= 300
    nchain  = 600
    nburn   = 200

    contwalkers = 100
    contchain   = 300
    contburn    = 150

elif MCMC_quality == 'lowqual':
    print("RUNNING WITH LOW QUALITY MCMC PARAMS \n")
    targappend='-lowqual'
    nwalkers= 50
    nchain  = 50
    nburn   = 10

    contwalkers = 50
    contchain   = 50
    contburn    = 10
elif MCMC_quality == "runtest":
    print("RUNNING RUN TEST\n")
    targappend='-runtest'
    nwalkers= 16
    nchain  = 5
    nburn   = 1

    contwalkers = 4
    contchain   = 5
    contburn    = 1

elif MCMC_quality == "longchain":
    print("RUNNING RUN TEST\n")
    targappend='-runtest'
    nwalkers= 240
    nchain  = 1000
    nburn   = 1

    contwalkers = 100
    contchain   = 500
    contburn    = 100

#===============================
#Read Directory for targets
url = targfolder + "/_dir.txt"
nosims, nogrades, targdirs  = read_dir(url)
#===============================
#Write record of MCMC properties to batch home folder
with open(targfolder+"/MCMCtrigger%s.txt" %targappend,'w') as f:
    f.write("\n Initial Continuum Search Params \n")
    f.write("contwalkers = %i \n" %contwalkers)
    f.write("Contchain = %i \n" % contchain)
    f.write("Contburn = %i \n" % contburn)

    f.write("\n Lag MCMC Params \n")
    f.write("nwalkers = %i \n" %nwalkers)
    f.write("nchain = %i \n" % nchain)
    f.write("nburn = %i \n" % nburn)
    f.write("maxlag  = %f \n" % laglimits[-1])
    f.close()
#===============================
for sim in range(nosims):
    print("Beginning run for sim %.2i" %sim)
    for grade in range(nogrades):
        print("Doing Javelin fits for data degredation level %.2i" %grade)

        #Get location of faked data
        targdir = targdirs[sim][grade]

        cont_url    = targdir + "/cont.dat"
        line1_url   = targdir + "/line1.dat"
        line2_url   = targdir + "/line2.dat"

        chain_url   = targdir + "/chain%s.dat" %targappend

        #================================
        #Load light curves into javelin-friendly objects
        cont_curve = get_data(cont_url)
        line1_curve= get_data(line1_url)
        line2_curve= get_data(line2_url)

        #Continuum Fitting
        print("Mapping continuum....")
        c_model = Cont_Model(cont_curve)
        c_model.do_mcmc(nwalkers = contwalkers, nchain = contchain, nburn = contburn, set_verbose = False)

        #Two Line Rmapping
        print("Attempting 2 line fit")
        line2_model = Rmap_Model(get_data([cont_url,line1_url,line2_url],names=["continuum","Line One", "Line Two"]))
        if fixed_widths:
            line2_model.do_mcmc(conthpd=c_model.hpd,    nwalkers=nwalkers,   nchain=nchain, nburn=nburn,  fchain=chain_url,
                            set_verbose = False,
                            laglimit=[laglimits,laglimits] ,
                            fixed = [1,1,1,0,1,1,0,1],    p_fix=[0,2,0,fixed_widths,1,0,fixed_widths,1])
        else:
            line2_model.do_mcmc(conthpd=c_model.hpd,    nwalkers=nwalkers,   nchain=nchain, nburn=nburn,  fchain=chain_url,
                                set_verbose = False,
                                laglimit=[laglimits,laglimits] ,
                                fixed = [1,1,1,1,1,1,1,1],    p_fix=[0,2,0,fixed_widths,1,0,fixed_widths,1])

        print("Done")
        print("=-=-=-=-=--=-=-=-=-=-=-")