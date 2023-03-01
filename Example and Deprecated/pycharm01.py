import numpy as np
import matplotlib.pylab as plt

'''
Python test workspace. Used to get head around Javelin
'''

#Import Zu Stuff
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model, Disk_Model, DPmap_Model

#Load .dat file into a "lightcurve" object. This is a wrapper for time-series.
c = get_data(["../examples/dat/continuum.dat"],names=["Continuum alone"])

#Generate a "continuum model" object from the zylc module
c_model = Cont_Model(c)
c_model.do_mcmc(set_verbose = True) #Perform mcmc

#Load spectroscopic lightcurve AND continuum lightcurve into single object
c_and_s = get_data(["../examples/dat/continuum.dat","../examples/dat/yelm.dat"],names=['Continuum for single', 'phot for single'])

c_and_s_model = Rmap_Model(c_and_s)
c_and_s_model.do_mcmc(conthpd=c_model.hpd) #Use "High probability domain" from continuum to do mcm to save on comp time

#Run sim with custom params, and also save results to target url
c_and_s_model.do_mcmc(nwalkers=100, nburn=100, nchain=100, fchain="chainout.dat")
'''
nwalkers    - No. MCMC chains
nburn       - No. its per chain to discard from output as "burn-in" steps
nchain      - No. its  to keep per chain. nburn+nchain = total steps per chain
'''

#Run sim with custom params, and also save results to target url
c_and_s_model.do_mcmc(lagtobaseline=0.3, laglimit='baseline')
'''
lagtobaseline - Ratio of maximum lag to total measurement time (baseline) to avoid high-tau issues
    .Default 30%
    .Not a hard limit. Values above are allowed but penalized heavily
Useful to use baseline lag to begin with, and then refine using these results
'''
c_and_s_model.do_mcmc(laglimit=[[100, 200],])
'''
laglimit    - Hardcoded bracketed allowable lags to test
Order of params in save:
    log(sigma), log(tau), {lag, width, scale}
'''

#Loading Saved Data for Chains
'''
LOAD A SAVED CHAIN:
some_model.load_chain("output.dat")
CUT CHAIN TO SOME LAG LIMIT
some_model.break_chain([[100, 200],])
UNDO THE LAG LIMIT CUTTING
some_model.restore_chain()
'''

print("Run complete")
