import numpy as np
import matplotlib.pylab as plt
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model, Disk_Model, DPmap_Model

'''
OzDesTwoLineTest.py

Test for Javelin understanding. Loads two lines for source

2925372393

v001

SUPERCEDED. DO NOT USE
'''
#================================
#Params
cont_url    = 'DataTest01/gband.dat'
line1_url   = 'DataTest01/CIV.dat'
line2_url   = 'DataTest01/MGII.dat'

chain_url = None

run_cont_first = True
run_one_lift    = True
run_baseline_first = True
#================================
print("==========================")
print("Starting")
cont_curve = get_data(cont_url)
line1_curve= get_data(line1_url)
line2_curve= get_data(line2_url)
print("Files Loaded")

#=============================================
if run_cont_first:
    print("Doing Cont. Model")
    c_model = Cont_Model(get_data(cont_url))
    c_model.do_mcmc()

#=============================================
print("Doing One Line")
line1_model = Rmap_Model(get_data([cont_url,line1_url],names=["continuum","Line One"]))
line1_model.do_mcmc(conthpd=c_model.hpd,nwalkers=200,nchain=500,nburn=150)

#=============================================
print("Doing Two Line")
line2_model = Rmap_Model(get_data([cont_url,line1_url,line2_url],names=["continuum","Line One", "Line Two"]))
line2_model.do_mcmc(conthpd=c_model.hpd,nwalkers=200,nchain=500,nburn=150)

#=============================================
print("Presenting Results")
RESULTS = line2_model.flatchain
lags_1 = RESULTS[:,2]
lags_2 = RESULTS[:,5]

plt.subplot(211)
plt.hist(line1_model.flatchain[:,2],bins=64,histtype='step',label="One Line lag 1",density=True,c='r')

plt.subplot(212)
plt.hist(lags_1,bins=64,histtype='step',label="Two Line lag 1",density=True,c='b')
plt.hist(lags_2,bins=64,histtype='step',label="Two Line lag 2",density=True,c='o')
plt.legend()
plt.grid()

plt.figure()
plt.subplot(211)
plt.title("CIV Only")
predcurve = line1_model.do_pred()
t = predcurve.jlist[0]
m = predcurve.mlist[0]
e = predcurve.mlist[1]
plt.plot(t,m,'k-')
plt.plot(t,m+e,'k--')
plt.plot(t,m-e,'k--')
#Data
plt.errorbar(cont_curve.jlist[0],cont_curve.mlist[0],yerr=cont_curve.elist[0],fmt='o',ms=4)
plt.grid()
plt.subplot(212)
predcurve = line2_model.do_pred()

plt.title("MGII and CIV")
t = predcurve.jlist[0]
m = predcurve.mlist[0]
e = predcurve.mlist[1]
plt.plot(t,m,'k-')
plt.plot(t,m+e,'k--')
plt.plot(t,m-e,'k--')
#Data
plt.errorbar(cont_curve.jlist[0],cont_curve.mlist[0],yerr=cont_curve.elist[0],fmt='o',ms=4)
plt.xlabel("Time (MJD)")
plt.grid()
plt.tight_layout()
plt.show()


print(" :)  ")