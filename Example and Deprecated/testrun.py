from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model

STAGES = 3
PLOTTING = True

if True:
    url_cont="2925344542_gBand.dat"
    url_line="2925344542_Hbeta_exp.dat"
    STAGES=min(STAGES,2)
else:
    url_cont="../examples/dat/continuum.dat"
    url_line="../examples/dat/yelm.dat"
    url_lineband="../examples/dat/yelmband.dat"

print("----------------")
if STAGES>=2:
    c = get_data([url_cont])
    cmod = Cont_Model(c)
    cmod.do_mcmc()
    if PLOTTING:
        cmod.show_hist(bins=100)

print("----------------")
if STAGES>=2:
    cy = get_data(["../examples/dat/continuum.dat", "../examples/dat/yelm.dat"])
    cymod = Rmap_Model(cy)
    print("Rmap done")

    cypred=cymod.do_pred(linhpd[1,:])
    print("mcmc done")
    if PLOTTING:
        try:
            cymod.show_hist(bins=100, lagbinsize=1)
            print("Hist done")
        except:
            print("Error on show hist")

print("----------------")
if STAGES>=3:
    cyb = get_data([url_cont, url_lineband])
    cybmod = Pmap_Model(cyb)
    cybmod.do_mcmc(conthpd=cmod.hpd)
    if PLOTTING:
        try:
            cybmod.show_hist()
            print("Hist done")
        except:
            print("Error on show hist")

print("\n\n\n ...done.")
