import numpy as np
import matplotlib.pylab as plt
import os
import chainconsumer
from reftable import *
from glob import glob
import acor
from copy import deepcopy as copy

'''
Chain_Analysis.py

HM 15/9

Changes
MISSINGNO to 19/9 - Various fixes

19/9
Truth values now read from sim_params
Support for single line analysis

#MISSINGNO
-Actually use truth values in plotting []
-Refactor into functions. Currently too unweildy
-Add continuum and burn chain plotting. Currently only runs for main chain
-Add "check for all grades in folder" functionality
-Fix contour boundaries


'''

#==============================================


def correl_time(X):
    '''
    Estimates the autocorrelation time of signal X using acor
    '''
    if X.shape[1]<X.shape[0]: X=X.T
    out=acor.acor(X)[0]
    return(out)

#==============================================

def do_hists(target_folder,grade,  verbose=False, mode='all'):

    #SETUP

    #Determine which chains to look for
    runtypes=runtypes_from_mode(mode)

    #------
    #Read dirfile
    nosims, no_subsamples,  targdirs = read_dir(target_folder+"/_dir.txt")
    #------
    #Make folder to store batch figures
    figfolder = target_folder+"/figs"
    if not os.path.exists(figfolder): os.makedirs(figfolder)
    #------
    #Look for simulation parameters
    
    param_url = target_folder + "/sim_params.npy"
    try:
        root_sim_params = np.load(param_url, allow_pickle=True)[0]
        if verbose: print("Loading root parameters from %s" % param_url)
    except:
        root_sim_params = None
        if verbose: print("Unable to locate root sim params for truth values")


    #Load MCMC Parameters
    MCMC_params = np.load(target_folder + "/MCMC_params-%s.npy" % grade, allow_pickle=True)[0]
    taumax = MCMC_params.laglimits[-1]

    for k,runtype in enumerate(runtypes):

        if verbose: print("Doing delay histograms for all chains of type %s" %runtype)

        #Make axes for histogram
        histfig, histaxs = plt.subplots(nosims, no_subsamples, figsize=(20,5))
        if not isinstance(histaxs, np.ndarray):
            histaxs = np.array([[histaxs]])
        elif histaxs.ndim==1:
            histaxs = np.array([histaxs])

        #Loop over all sims and subsamplings
        for i in range(nosims):
            for j in range(no_subsamples):

                #Load Folder
                folder = targdirs[i][j]

                chain_url = folder + "/chain-%s-%s.dat" %(runtype, grade)
                if not os.path.exists(chain_url):
                    print("couldnt find file %s" % chain_url)
                    continue  # Safety check to avoid loading bad chain
                else:
                    if verbose: print("\tDoing Histogram for %s" %chain_url)

                #Load Chain
                CHAIN = np.loadtxt(chain_url)

                #Try to load truth values
                param_url = folder + "/sim_params.npy"
                try:
                    sim_params = np.load(param_url, allow_pickle=True)[0]
                    if verbose: print("Loading fixed parameters from %s" % param_url)
                except:
                    sim_params = None
                    if verbose: print("Unable to locate sim params, defaulting to root params for sim %s" %folder)
                if sim_params==None:
                    if root_sim_params==None:
                        del_truth=None
                    else:
                        del_truth=[root_sim_params.delay1,root_sim_params.delay2]
                else:
                    del_truth=[sim_params.delay1,sim_params.delay2]   
                        
                #----------------------
                #Do hist plots

                ax = histaxs[i][j]

                if runtype == 'twoline':
                    TAU_1 = CHAIN[:, 2]
                    TAU_2 = CHAIN[:, 5]

                    ax.hist(TAU_1, histtype="step", density=True, bins=64, color = line1_color)
                    ax.hist(TAU_2, histtype="step", density=True, bins=64, color = line2_color)
                    

                    if not del_truth==None:
                        ax.axvline(del_truth[0], c=line1_color, linewidth = 2 )
                        ax.axvline(del_truth[1], c=line2_color, linewidth = 2 )

                elif runtype == 'line1':
                    TAU_1 = CHAIN[:, 2]

                    ax.hist(TAU_1, histtype="step", density=True, bins=64, color = line1_color)

                    if not del_truth==None:
                        ax.axvline(del_truth[0], c=line1_color, linewidth = 2 )

                elif runtype == 'line2':
                    TAU_2 = CHAIN[:, 2]

                    ax.hist(TAU_2, histtype="step", density=True, bins=64, color = line2_color)

                    if not del_truth==None:
                        ax.axvline(del_truth[1], c=line2_color, linewidth = 2 )
                ax.set_xlim([0, taumax])
                #--------------------------------------------

                #Plot seasonal lines
                for t in range(0, taumax, 180):
                    ax.axvline(t, ls='--', alpha=0.5, lw=2, color='grey')

                #Set Axis Limits
                if i != range(nosims)[-1]:
                    ax.set_xticks([])
                else:
                    ax.set_xticks(range(0, taumax, 180))
                ax.set_yticks([])

        #Add titles and axis labels
        for j in range(no_subsamples):
            title = targdirs[0][j]
            title=title[::-1][:title[::-1].index("/")][::-1]
            histaxs[0][j].title.set_text(title)

            histaxs[-1][j].set_xlabel("Delay (Days)")
        if runtype=='twoline':
            histfig.legend(["Line 1 Delay", "Line 2 Delay"], loc='right')
        #histfig.tight_layout()

        hist_url = figfolder + "/Histograms-%s-%s.png" %(runtype,grade)
        if verbose: print("saving to %s" %hist_url)
        histfig.savefig(hist_url, format='png')
        plt.close(histfig)

#===================
#Contour Functions

def do_chain_contours(folder, grade, sim_params=None, MCMC_params=None,  mode='all', verbose=False):
    if verbose: print("\t\t Getting Contours")


    runtypes = runtypes_from_mode(mode)
    for runtype in runtypes:

        if runtype == 'twoline':
            c=twoline_color
        elif runtype=='line1':
            c=line1_color
        elif runtype=='line2':
            c=line2_color

        #Find Chain URL
        chain_url = folder + "/chain-%s-%s.dat" % (runtype, grade)
        if not os.path.exists(chain_url):
            print("couldnt find file %s" % chain_url)
            continue  # Safety check to avoid loading bad chain
        else:
            if verbose: print("\tDoing Contour Plots for %s" % chain_url)

        # Load Chain
        CHAIN = np.loadtxt(chain_url)

        # Load fixed parameters
        paramnames = ["$ln|\sigma_c|$", "$ln|t_c|$", "$\Delta t_1$", "$w_1$", "$\sigma_1$", "$\Delta t_2$", "$w_2$","$\sigma_2$"]
        fixed        = MCMC_params.get_fixed_array(runtype=runtype)
        truth_params = sim_params.get_truth(runtype=runtype)

        #Trim chain, truth values and param names to the required size
        cc_chain = np.delete(CHAIN, [i for i in range(len(fixed)) if fixed[i] == 0], 1) #Trim the chain fixed chain columns
        truth_params = [truth for i, truth in enumerate(truth_params) if fixed[i] == 1]
        if runtype == 'line1': paramnames = paramnames[:5]
        if runtype == 'line2': paramnames = paramnames[:2] + paramnames[5:]
        paramnames = [name for i, name in enumerate(paramnames) if fixed[i] == 1]

        #Do chainconsumer contours for entire set
        main_cc = chainconsumer.ChainConsumer().add_chain(cc_chain, parameters=paramnames)
        main_cc.configure(colors=[c],sigmas=contour_sigmas)
        cfig = main_cc.plotter.plot(truth=truth_params) #[DISABLED]
        #cfig = main_cc.plotter.plot()
        cfig.tight_layout()
        cfig.savefig(folder + "/contours_all-%s-%s.png" % (runtype, grade), format='png')
        plt.close(cfig)

        #Do delays only plot if twoline
        if runtype == 'twoline':
            #Make Chainconsumer plot
            delay_cc = chainconsumer.ChainConsumer().add_chain(np.vstack([CHAIN[:, 2], CHAIN[:, 5]]).T, parameters=["$\Delta t_1$", "$\Delta t_2$"])
            delay_cc.configure(colors=[twoline_color], sigmas=contour_sigmas)
            cfig = delay_cc.plotter.plot(truth=[sim_params.delay1, sim_params.delay2]) #[DISABLED]
            #cfig = delay_cc.plotter.plot(extents=[(0,800),(0,800)])

            #Arrange and save figure
            cfig.tight_layout()
            cfig.savefig(folder + "/contours_delaysonly-twoline-%s.png" % (grade), format='png')
            plt.close(cfig)


def do_continuum_contours(folder, grade, sim_params=None, MCMC_params=None, verbose=False):
    if verbose: print("\t\t Getting Continuum Contours")

    #Find Chain URL
    chain_url = folder + "/chain-cont-%s.dat" % (grade)
    if not os.path.exists(chain_url):
        print("couldnt find file %s" % chain_url)
        return
    else:
        if verbose: print("\tDoing Continuum Contours for %s" % chain_url)

    # Load Chain
    CHAIN = np.loadtxt(chain_url)

    # Load fixed parameters
    paramnames = ["$ln|\sigma_c|$", "$ln|t_c|$"]
    fixed        = MCMC_params.get_fixed_array()[:2]
    truth_params = sim_params.get_truth()[:2]
    if sum(fixed)==0:
        print("Continuum parameters fixed for chain %s. Can't do a contour." %chain_url)
        return

    #Do chainconsumer contours for entire set
    cc = chainconsumer.ChainConsumer().add_chain(CHAIN, parameters=paramnames)
    cc.configure(colors=[continuum_color],sigmas=contour_sigmas)
    cfig = cc.plotter.plot(truth=truth_params) #[DISBLED]
    #cfig = cc.plotter.plot()
    cfig.tight_layout()
    cfig.savefig(folder + "/contours_continuum-%s.png" % (grade), format='png')
    plt.close(cfig)

def do_comparison_contours(folder, grade, sim_params=None, MCMC_params=None, verbose=False):

    chain_urls = chain_url_twoline, chain_url_line1, chain_url_line2 =   [folder + "/chain-%s-%s.dat" %(runtype,grade) for runtype in ['twoline','line1','line2']]
    for url in [chain_url_twoline, chain_url_line1, chain_url_line2]:
        if not os.path.exists(url):
            print("couldnt find file %s in comparison contours. Skipping" % url)
            return

    #Main Contours
    main_chain = chainconsumer.ChainConsumer()
    delay_chain=chainconsumer.ChainConsumer()

    #Load each chain
    CHAIN_twoline = np.loadtxt(chain_url_twoline)
    CHAIN_line1   = np.loadtxt(chain_url_line1)
    CHAIN_line2   = np.loadtxt(chain_url_line2)

    CHAIN_twoline_lags = np.vstack([CHAIN_twoline[:, 2], CHAIN_twoline[:, 5]]).T
    CHAIN_oneline_lags = np.vstack([CHAIN_line1[:, 2], CHAIN_line2[:, 2]]).T

    truth_params = sim_params.get_truth(runtype = 'twoline')
    fixed        = MCMC_params.get_fixed_array(runtype='twoline')
    truth_params = [truth for i, truth in enumerate(truth_params) if fixed[i] == 1]

    #(Messily) load chains into the chain consumers
    paramnames  = ["$ln|\sigma_c|$", "$ln|t_c|$", "$\Delta t_1$", "$w_1$", "$\sigma_1$", "$\Delta t_2$", "$w_2$","$\sigma_2$"]

    CHAIN_twoline = np.delete(CHAIN_twoline, [i for i in range(len(fixed)) if fixed[i] == 0],1)
    paramnames_temp = [name for i, name in enumerate(paramnames) if fixed[i] == 1]
    main_chain.add_chain(CHAIN_twoline, parameters=paramnames_temp, name="Two Lines")

    fixed = MCMC_params.get_fixed_array(runtype = 'line1')
    CHAIN_line1 = np.delete(CHAIN_line1, [i for i in range(len(fixed)) if fixed[i] == 0],1)
    paramnames_temp = paramnames[:5]
    paramnames_temp = [name for i, name in enumerate(paramnames_temp) if fixed[i] == 1]
    main_chain.add_chain(CHAIN_line1, parameters=paramnames_temp, name="Line 1 Only")

    fixed = MCMC_params.get_fixed_array(runtype = 'line2')
    CHAIN_line2 = np.delete(CHAIN_line2, [i for i in range(len(fixed)) if fixed[i] == 0],1)
    paramnames_temp = paramnames[:2]+paramnames[5:]
    paramnames_temp = [name for i, name in enumerate(paramnames_temp) if fixed[i] == 1]
    main_chain.add_chain(CHAIN_line2, parameters=paramnames_temp,name="Line 2 Only")

    delay_chain.add_chain(CHAIN_twoline_lags, parameters=["$\Delta t_1$", "$\Delta t_2$"],  name="Both lines")
    delay_chain.add_chain(CHAIN_oneline_lags, parameters=["$\Delta t_1$",  "$\Delta t_2$"], name="Independent lines")

    #Do plots
        # All Parameters
    main_chain.configure(colors=[twoline_color,line1_color,line2_color],linestyles=["-", "--","--"], shade_alpha=[.6,.8,.8],sigmas=contour_sigmas)
    mainplot = main_chain.plotter.plot(truth=truth_params)#[DISABLED]
    #mainplot = main_chain.plotter.plot() 
    mainplot.tight_layout()
    mainplot.savefig(folder+"/contours_combined_all-%s" %grade)
    plt.close(mainplot)

        #Delay delay comparison
    delay_chain.configure(colors=[twoline_color,'cyan'],linestyles=['-','-'],shade_alpha=[.6,.6], legend_kwargs={"loc": "upper left", "fontsize": 10},legend_location=(0, 0),sigmas=contour_sigmas)
    delayplot = delay_chain.plotter.plot(truth=[sim_params.delay1,sim_params.delay2])#[DISABLED]
    #delayplot = delay_chain.plotter.plot() 
    delayplot.tight_layout()
    delayplot.savefig(folder+"/contours_combined_delaysonly-%s" %grade)
    plt.close(delayplot)

#===================
def do_acceptance_ratios(folder,grade,mode='all',verbose=False):
    runtypes = runtypes_from_mode(mode)

    out_url = folder + "/acceptance-%s.dat" % (grade)
    out_file=open(out_url,'w')

    out_file.write("\t%s\t%s\t%s\n" % ("N", "Accept", "AR"))
    for runtype in runtypes:
        chain_url = folder + "/chain-%s-%s.dat" % (runtype, grade)
        if not os.path.exists(chain_url):
            continue  # Safety check to avoid loading bad chain

        # Load Chain
        CHAIN = np.loadtxt(chain_url)

        tot_points = CHAIN.shape[0]
        unique_points = len(np.unique(CHAIN, axis=0))
        AR = unique_points / tot_points

        out_file.write("%s\t%i\t%i\t%f\n" %(runtype,tot_points,unique_points,AR))
    out_file.close()

def do_correl_times(folder, grade, mode='all',verbose=False):
    runtypes = runtypes_from_mode(mode)

    out_url = folder + "/correltime-%s.dat" % (grade)
    out_file=open(out_url,'w')

    out_file.write("%s\n" % ("tau"))
    for runtype in runtypes:
        chain_url = folder + "/chain-%s-%s.dat" % (runtype, grade)
        if not os.path.exists(chain_url):
            continue  # Safety check to avoid loading bad chain
        CHAIN = np.loadtxt(chain_url)

        # Load Chain
        try:
            tau = correl_time(CHAIN)
            out_file.write("%s\t%f\n" % (runtype, tau))
        except:
            out_file.write("%s\t%s\n" % (runtype, "NaN"))


    out_file.close()

# ===================


def batch_analysis(target_folder,  grade, histograms=True, contours = True, correltimes=True, acceptance_ratios = True, verbose =False, mode='all'):
    '''
    batch_analysis(target_folder, grade)

    Generates images and such for all targets in a batch

    Inputs
        target_folder       str             URL of the batch to perform analysis for
        grade               str             Appendix for files to read from / write to
    Additional Arguments
        contours            bool    True    If True, generates chainconsumer parameter plots
        correltimes         bool    True    If True, estimates the correlation time of the chains
        acceptance_ratios   bool    True    If True, calculates the acceptance ratio of the chain
        verbose             bool    False   If True, prints additional progress tracking
        mode                str     all     Which chains to do analysis for:
                                                twoline - Default chain file only
                                                line1   - line1 file only
                                                line2   - line2 file only
                                                both    - both line 1 and line 2
                                                all     - all files
    '''

    #SETUP

    #Determine which chains to look for
    runtypes=runtypes_from_mode(mode)

    #------
    #Read dirfile
    nosims, no_subsamples,  targdirs = read_dir(target_folder+"/_dir.txt")

    #------
    #Make folder to store batch figures
    figfolder = target_folder+"/figs"
    if not os.path.exists(figfolder): os.makedirs(figfolder)

    #------
    #Look for simulation parameters
    param_url = target_folder + "/sim_params.npy"
    try:
        root_sim_params = np.load(param_url, allow_pickle=True)[0]
        if verbose: print("Loading fixed parameters from %s" % param_url)
    except:
        root_sim_params = None
        if verbose: print("Unable to locate sim params for truth values. Loading default")

    #Load MCMC Parameters
    MCMC_params = np.load(target_folder + "/MCMC_params-%s.npy" % grade, allow_pickle=True)[0]

    #=====================================

    if histograms: do_hists(target_folder=target_folder, grade=grade, verbose=verbose, mode=mode)

    for k,runtype in enumerate(runtypes):
        if verbose: print("Doing analysis for all chains of type %s" %runtype)

        taumax=MCMC_params.laglimits[-1]

        if verbose: print("Settup done, beginning analysis")

        for i in range(nosims):
            for j in range(no_subsamples):

                #Load Folder
                folder = targdirs[i][j]

                chain_url = folder + "/chain-%s-%s.dat" %(runtype, grade)
                chain_url=chain_url.replace('--','-')
                if not os.path.exists(chain_url):
                    print("couldnt find file %s" % chain_url)
                    continue  # Safety check to avoid loading bad chain
                else:
                    if verbose: print("\tDoing Plots for %s" %chain_url)

                #Try to load truth values
                param_url = folder + "/sim_params.npy"
                try:
                    sim_params = np.load(param_url, allow_pickle=True)[0]
                    if verbose: print("Loading fixed sim specific parameters from %s" % param_url)
                except:
                    sim_params = None
                    if verbose: print("Unable to locate sim params for truth values at %s. Using Root" %param_url)
                    
                if sim_params==None: sim_params=copy(root_sim_params)

                #Load Chain
                if not( contours or correltimes or acceptance_ratios): continue
                CHAIN = np.loadtxt(chain_url)

                #ChainConsumer Contours
                if contours:
                    do_chain_contours(folder=folder,grade=grade,sim_params=sim_params,MCMC_params=MCMC_params,mode=mode,verbose=verbose)

                    if mode=='all':
                        do_comparison_contours(folder=folder, grade=grade, sim_params=sim_params, MCMC_params=MCMC_params, verbose=verbose)
                    do_continuum_contours(folder=folder, grade=grade, sim_params=sim_params, MCMC_params=MCMC_params, verbose=verbose)

                #Get correlation Times
                if correltimes:
                    if verbose: print("\t\t Getting Correlation time")
                    do_correl_times(folder=folder, grade=grade, mode=mode, verbose=verbose)

                #Get Acceptance Ratios
                if acceptance_ratios:
                    if verbose: print("\t\t Getting acceptance ratios")
                    do_acceptance_ratios(folder=folder, grade=grade, mode=mode, verbose=verbose)

if __name__=="__main__":
    batch_analysis("./Data/fakedata/Batch_Test",  grade = 'runtest')

    print("Done")
