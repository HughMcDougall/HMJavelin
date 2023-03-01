import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from javelin.predict import PredictSignal, PredictRmap, generateLine, generateError, PredictSpear
from javelin.lcio import *
from javelin.zylc import LightCurve, get_data
from javelin.lcmodel import Cont_Model, Rmap_Model, Pmap_Model
from scipy import stats
from astropy.io import fits
import random as rn
import pandas as pd
import QOL_funcs as qol
import sys, os
import time
from matplotlib import rc

rc('text', usetex=False)


def jav_rec(name, expected_lag, walkers=200, chains=500, line='CIV', f_in='../Processed_LC_Y6/', f_out='obs_data/JAV/'):
    tau_lim = 3 * expected_lag
    if tau_lim >= 1850:
        tau_lim = 1850
    elif tau_lim <= 650:
        tau_lim = 650

    dir_figs = f_out + 'figs/'
    if not os.path.exists(dir_figs):
        try:
            os.makedirs(dir_figs)
        except:
            pass
    dir_hist_txt = f_out + 'pdf_txt/'
    if not os.path.exists(dir_hist_txt):
        os.makedirs(dir_hist_txt)

    object_dir = f_in + str(name)

    javdata1 = get_data([object_dir + '_gBand.txt'], names=["continuum"])

    javdata2 = get_data([object_dir + '_gBand.txt', object_dir + '_' + line + '_exp.txt'], names=["Continuum", "Line"])
    # javdata2 = get_data([object_dir+'_gBand.txt',object_dir+'_'+line+'.txt'], names=["Continuum", "Line" ])
    cont = Cont_Model(javdata1)

    cont.do_mcmc(set_prior=False, nwalkers=200, nburn=200, nchain=200, threads=1)

    rmap1 = Rmap_Model(javdata2)

    # rmap1.do_mcmc(conthpd=cont.hpd,nwalkers=walkers, nburn=150, nchain=chains, laglimit=[[0, tau_lim]],threads=1)
    rmap1.do_mcmc(conthpd=cont.hpd, nwalkers=walkers, nburn=150, nchain=chains, laglimit=[[0, tau_lim]],
                  fixed=[1, 1, 1, 0, 1], p_fix=[10, 10, 10, 30, 10], threads=1)

    rmap1.do_pred(fpred=dir_figs + str(name) + '.txt',
                  dense=20)  # .plot(set_pred=True,obs=javdata2,figout=dir_figs+str(name),figext='png', names=["Continuum", "Line" ])

    pdf_out = dir_hist_txt + str(name) + 'pdf.txt'
    out_chain = open(pdf_out, 'w')

    out_chain.writelines("%s\n" % place for place in rmap1.flatchain[:, 2])

    out_chain.close()
    return rmap1.flatchain[:, 2]


def jav_rec_sims(name, lag_fracs, lag_exp, num_reals, walkers=150, chains=500, files='sims_data/'):
    out_lags_all = open(files + str(name) + '/lags_jav.txt', 'w')

    tau_lim = 3 * lag_exp
    if tau_lim >= 1850:
        tau_lim = 1850

    for i in range(len(lag_fracs)):

        base_dir = files + str(name) + '/' + str(lag_fracs[i]) + '/'
        out_lags_fracs = open(base_dir + '/lags_jav_' + str(lag_fracs[i]) + '.txt', 'w')

        dir_figs = base_dir + 'fig_jav/'
        if not os.path.exists(dir_figs):
            os.makedirs(dir_figs)

        dir_hist_txt = base_dir + 'pdfs_jav/'
        if not os.path.exists(dir_hist_txt):
            os.makedirs(dir_hist_txt)

        dir_txt = base_dir + 'txt/'

        for j in range(num_reals):

            object_dir = dir_txt + str(name) + '_' + str(j) + '_' + str(lag_fracs[i])

            javdata1 = get_data([object_dir + '_cont.txt'], names=["continuum"])

            javdata2 = get_data([object_dir + '_cont.txt', object_dir + '_line.txt'], names=["Continuum", "Line"])

            cont = Cont_Model(javdata1)

            cont.do_mcmc(set_prior=False, nwalkers=50, nburn=50, nchain=50, threads=1)

            rmap1 = Rmap_Model(javdata2)

            rmap1.do_mcmc(conthpd=cont.hpd, nwalkers=walkers, nburn=50, nchain=chains, laglimit=[[0, tau_lim]],
                          threads=1, fixed=[1, 1, 1, 0, 1], p_fix=[10, 10, 10, 30, 10])

            name_out = str(name) + '_' + str(lag_fracs[i]) + '_' + str(j)
            if j < 10:
                rmap1.do_pred(fpred=dir_figs + str(name) + '.txt', dense=20)
                # .plot(set_pred=True,obs=javdata2,figout=dir_figs+ name_out, figext='png', names =['Continuum','Line'])

                plt.close('all')
            out_chain = open(dir_hist_txt + name_out + '.txt', 'w')

            out_chain.writelines("%s\n" % place for place in rmap1.flatchain[:, 2])
            out_chain.close()

            lag, err = qol.Peak_Mean_AD(rmap1.flatchain[:, 2], lag_exp)

            out_lags_all.write(str(lag_fracs[i]) + '\t' + str(lag) + '\t' + str(err) + '\n')

            out_lags_fracs.write(str(lag) + '\t' + str(err) + '\n')
        out_lags_fracs.close()
    out_lags_all.close()


def jav_rec_random(name, lag_fracs, lag_exp, num_reals, walkers=150, chains=500, files='sims_data/'):
    out_lags_all = open(files + str(name) + '/lags_jav_rand.txt', 'w')

    tau_lim = 3 * expected_lag
    if tau_lim >= 1850:
        tau_lim = 1850
    elif tau_lim <= 650:
        tau_lim = 650

    for i in range(len(lag_fracs)):

        base_dir = files + str(name) + '/' + str(lag_fracs[i]) + '/'
        out_lags_fracs = open(base_dir + '/lags_jav_rand' + str(lag_fracs[i]) + '.txt', 'w')

        dir_figs = base_dir + 'fig_jav_rand/'
        if not os.path.exists(dir_figs):
            os.makedirs(dir_figs)

        dir_hist_txt = base_dir + 'pdfs_jav_rand/'
        if not os.path.exists(dir_hist_txt):
            os.makedirs(dir_hist_txt)

        dir_txt = base_dir + 'txt/'

        for j in range(num_reals):

            object_dir = dir_txt + str(name) + '_' + str(j) + '_' + str(lag_fracs[i])

            javdata1 = get_data([object_dir + '_randcont.txt'], names=["continuum"])

            javdata2 = get_data([object_dir + '_randcont.txt', object_dir + '_randline.txt'],
                                names=["Continuum", "Line"])

            cont = Cont_Model(javdata1)

            cont.do_mcmc(set_prior=False, nwalkers=50, nburn=50, nchain=50, threads=1)

            rmap1 = Rmap_Model(javdata2)

            rmap1.do_mcmc(conthpd=cont.hpd, nwalkers=walkers, nburn=50, nchain=chains, laglimit=[[0, tau_lim]],
                          threads=1, fixed=[1, 1, 1, 0, 1], p_fix=[10, 10, 10, 30, 10])

            name_out = str(name) + '_' + str(lag_fracs[i]) + '_' + str(j)
            if j < 10:
                rmap1.do_pred(fpred=dir_figs + str(name) + '.txt', dense=20)
                # .plot(set_pred=True,obs=javdata2,figout=dir_figs+ name_out, figext='png', names =['Continuum','Line'])

                plt.close('all')
            out_chain = open(dir_hist_txt + name_out + '.txt', 'w')

            out_chain.writelines("%s\n" % place for place in rmap1.flatchain[:, 2])
            out_chain.close()


'''
#javdata2 = get_data([object_dir+'_rBand.txt',object_dir+'_iBand.txt',object_dir+'_'+line+'.txt'], names=["Continuum (r)","Continuum (i", "Line" ])

#javdata2 = get_data([object_dir+'_gBand.txt',object_dir+'_rBand.txt',object_dir+'_iBand.txt',object_dir+'_'+line+'.txt'], names=["Continuum (g)","Continuum (r)","Continuum (i)", "Line" ])

rmap1.do_mcmc(conthpd=cont.hpd,nwalkers=walkers, nburn=50, nchain=chains, laglimit=[[-30,30],[0, tau_lim]],fixed=[1,1,1,0,1,1,0,1],p_fix=[10,10,10,30,10,10,30,10])


rmap1.do_pred(dense=20).plot(set_pred=True,obs=javdata2,figout=dir_figs+str(name),figext='png', names=["Continuum (r)","Continuum (i)","Line" ])


rmap1.do_mcmc(conthpd=cont.hpd,nwalkers=walkers, nburn=200, nchain=chains, laglimit=[[0,30],[0,30],[0, tau_lim]],fixed=[1,1,1,0,1,1,0,1,1,0,1],p_fix=[10,10,10,30,10,10,30,10,10,30,10])


rmap1.do_pred(dense=20).plot(set_pred=True,obs=javdata2,figout=dir_figs+str(name),figext='png', names=["Continuum (g)","Continuum (r)","Continuum (i)", "Line" ])


pdf_out = dir_hist_txt+str(name)+'pdf_r.txt'
out_chain = open(pdf_out,'w')

out_chain.writelines("%s\n" % place for place in rmap1.flatchain[:,2])
out_chain.close()

pdf_out = dir_hist_txt+str(name)+'pdf_i.txt'
out_chain = open(pdf_out,'w')

out_chain.writelines("%s\n" % place for place in rmap1.flatchain[:,5])
out_chain.close()

pdf_out = dir_hist_txt+str(name)+'pdf.txt'
out_chain = open(pdf_out,'w')

out_chain.writelines("%s\n" % place for place in rmap1.flatchain[:,8])
'''



