from reftable import *
from Datagen import simgen_batch
from MCMC_Utils import MCMCrun_batch
from Chain_Analysis import batch_analysis
from copy import deepcopy as copy


'''
runtime01.py

Trigger for MCMC test runs

HM 20/9
'''

sim=sim_params()
sim.delay1=100
sim.delay2=250
sim.cont_tau=130*3

#====================================
batch_name  = "uncorreltest-2"
grade       = ultrahighqual
nosims      = 3
sample_types=[ClearSignal, JavelinExample, source_B1, source_A1, source_B6, source_A7]
#sample_types=[ClearSignal, JavelinExample, source_A1]
#====================================
batch_folder= "./Data/fakedata/" + batch_name

print("Starting Simulations")
#simgen_batch(SAMPLING_PARAMS=sample_types, nosims = nosims, targfolder=batch_folder,    sim_params=sim, verbose=True)
print("Doing MCMC")
MCMCrun_batch(targfolder = batch_folder, MCMC_params=grade,  verbose=True, mode = 'all',       output_burn=True,   output_cont=True, output_logp  = True, skipexisting=True)
print("Doing Figs")
batch_analysis(batch_folder,grade.name, histograms = True, contours=True, correltimes=False, acceptance_ratios=False,  verbose=True, mode='all')
print("Done.")
