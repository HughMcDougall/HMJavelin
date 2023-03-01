from javelin.zylc import get_data
from reftable import *
from warnings import warn
from javelin.zylc import get_data
from javelin.lcmodel import Cont_Model, Rmap_Model
import os

'''
MCMC_Utils.py
Functions for running javelin's MCMC chains more easily

Changes
18/9
Added arguments to output burn and continuum chains

19/9
Added width limit (default 60 days) to all MCMC runs
Updated documentation
Added single line functionality
Changed default values in p_fix to 10 for all but the tophat width, which remains at 30. Hopefully this stops divergence
Can now otput continuum and logs

20/9
Fixed some errors in loading default params

21/9
Can now skip existing chains to "pick up" an old run
'''

def MCMCrun_single(targdir, MCMC_params = MCMC_grade(), mode = 'twoline', datasource = None, savefolder=None, verbose = False, truth_value_params = None, output_burn=False, output_cont=False, output_logp=False, skipexisting=True):
    '''
    MCMCrun_single(targdir, MCMC_params = MCMC_grade(), datasource = None, savefolder=None)

    Required Arguments
        targdir     str             URL of the folder to get / save to

    Additional Arguments
        Name                Type        Default
        mode                str         twoline     What type of RM to do
                                                        -twoline (default) - Do both lines simultaneously
                                                        -line1  Do RM for only line 1
                                                        -line2  Do RM for only line 2
                                                        -both   Do RM for both lines seperately
                                                        -all    Do both lines and also the twoline
        MCMC_params         MCMC_grade  None        See reftable.py for pre-made MCMC grades and their properties
        datasource          datasource  None        Override data source to compare model to. If not set, will look in targdir for valid .dat files
        savefolder          str         None        Override chain output location. If not set, will save to targdir
        verbose             bool        False       If True, will progress statements
        truth_value_params  simparams   None        Truth values to fix parameters at. If not found, will search targdir for a valid .npy file containing an object
        output_continuum    bool        False       If True and MCMC_params.do_continuum==True, will output the continuum MCMC chain file
        output_burn         bool        False       If True, will output burn steps in the MCMC chain to its own file
        output_logp         bool        False       If True, will also output the log likelihood of the main chain to a file
        skip_existing       bool        False       If True, will not run MCMC for existing files. NOT WORKING CURRENTLY


    '''

    #Load MCMC Parameters
    nwalkers    = MCMC_params.nwalkers
    nchain=nburn= MCMC_params.nchain

    contwalkers = MCMC_params.contwalkers
    contchain   = MCMC_params.contchain
    contburn    = MCMC_params.contburn

    laglimits   = MCMC_params.laglimits
    widlimits   = MCMC_params.widlimits

    # Try to load run name from MCMC params. If none defined, use no appendix to filenames
    if MCMC_params.name==None:
        targappend=''
        warn("Warning! MCMC run name provided in MCMC_params. Output chain may over ride other runs")
    else:
        targappend = '-' + MCMC_params.name

    # Get location of faked data
    if datasource == None:
        #If no datasourve overide, look in targdir for source files
        cont_url  = targdir + "/cont.dat"
        line1_url = targdir + "/line1.dat"
        line2_url = targdir + "/line2.dat"

    #Create list of run type to use (one line, two line etc)
    runtypes=[]
    if mode == 'twoline' or mode=='all':
        runtypes.append('twoline')
    if mode == 'both' or mode == 'line1' or mode == 'all':
        runtypes.append('line1')
    if mode == 'both' or mode == 'line2' or mode == 'all':
        runtypes.append('line2')

    #If no override on output, use targdir
    if savefolder ==None: savefolder=targdir
    chain_urls =   [savefolder + "/chain-%s%s.dat" %(runtype,targappend) for runtype in runtypes]

    #Get alternate save URLS
    if output_logp:
        logp_urls = [savefolder + "/logp-%s%s.dat" %(runtype,targappend) for runtype in runtypes]
    else:
        logp_urls = [None]*len(runtypes)

    if output_burn:
        burn_urls = [savefolder + "/burn-%s%s.dat" %(runtype,targappend) for runtype in runtypes]
    else:
        burn_urls = [None]*len(runtypes)
    if output_cont:
        contchain_url = savefolder + "/chain-cont%s.dat" % targappend
    else:
        contchain_url = None

    # ================================
    # Load light curves into javelin-friendly objects
    cont_curve  = get_data(cont_url)
    line1_curve = get_data(line1_url)
    line2_curve = get_data(line2_url)

    #Look for simulation parameters to fix values
    if truth_value_params==None:
        truth_value_params=sim_params()
        warn('unable to load truth values from sim_params. Assuming defaults')

    #Use to generated fixed values
    fix   = MCMC_params.get_fixed_array()
    p_fix = MCMC_params.p_fix
    truevals = truth_value_params.get_truth()

    #If no values are fixed, set p_fix to Nonetype to let Javelin to initialization itself
    if sum(MCMC_params.get_fixed_array()) == 0:
        fix,p_fix,fix_1,p_fix_1,fix_2,p_fix_2     =   [None]*6
    else:
        #otherwise, fill out p_fix with true fixed values...
        for i, f in enumerate(fix):
            if f == 0: p_fix[i] = truevals[i]
        #trim the fixed values for lines to the correct size
        fix_1   = [f for f,z in zip(fix, [1,1,1,1,1,0,0,0]) if z==1]
        p_fix_1 = [p for p,z in zip(p_fix, [1,1,1,1,1,0,0,0]) if z==1]
        fix_2   = [f for f,z in zip(fix, [1,1,0,0,0,1,1,1]) if z==1]
        p_fix_2 = [p for p,z in zip(p_fix, [1,1,0,0,0,1,1,1]) if z==1]

    if verbose:
        print("Beginning MCMC")
        print("Fixed and Free params:")
        for name,a,b in zip(['ln_sig','ln_tau','delay1','width','amp1','delay2','width2','amp2'],fix,p_fix):
            print(name,'\t',a,'\t',b)

    # Continuum Fitting
    if MCMC_params.do_continuum or output_cont:
        c_model = Cont_Model(cont_curve)
        if verbose: print("Doing Continuum MCMC")

        #Skip existing chains
        if os.path.exists(contchain_url) and skipexisting:
            c_model.load_chain(contchain_url)
        else:
            c_model.do_mcmc(nwalkers=contwalkers, nchain=contchain, nburn=contburn, set_verbose=verbose, fixed = fix[0:2], p_fix = p_fix[0:2], fchain=contchain_url)

    # Two Line Rmapping
    if verbose: print("Doing RM MCMC")
    models=[]
    fixes=[]
    p_fixes=[]
    if mode == 'twoline' or mode=='all':
        twoline_model = Rmap_Model(get_data([cont_url, line1_url, line2_url], names=["Continuum", "Line One", "Line Two"]))
        models.append(twoline_model)
        fixes.append(fix)
        p_fixes.append(p_fix)

    if mode == 'both' or mode == 'line1' or mode == 'all':
        line1_model  = Rmap_Model(get_data([cont_url, line1_url], names=["Continuum", "Line One"]))
        models.append(line1_model)
        fixes.append(fix_1)
        p_fixes.append(p_fix_1)

    if mode == 'both' or mode == 'line2' or mode == 'all':
        line2_model  = Rmap_Model(get_data([cont_url, line2_url], names=["Continuum", "Line Two"]))
        models.append(line2_model)
        fixes.append(fix_2)
        p_fixes.append(p_fix_2)

    for model,chain_url,burn_url,logp_url,fixed,p_fixed in zip(models, chain_urls, burn_urls, logp_urls, fixes, p_fixes):

        #Skip current chain if it exists
        if os.path.exists(chain_url) and skipexisting:
            if verbose: print("Chain %s already exists. Skipping." %chain_url)
            continue

        if verbose: print("Doing MCMC for %s" %chain_url.removeprefix(targdir))
        #Make lag and width limits the correct size
        if len(fixed)==8:
            ll = [laglimits,laglimits]
            wl = [widlimits,widlimits]
        else:
            ll = [laglimits]
            wl = [widlimits]

        #Run MCMC on the model. Use continuum HPD if available.
        if MCMC_params.do_continuum==True:
            model.do_mcmc(conthpd=c_model.hpd,
                                nwalkers=nwalkers, nchain=nchain, nburn=nburn,
                                fchain=chain_url,
                                fburn=burn_url, flogp=logp_url,
                                set_verbose=verbose,
                                laglimit=ll,
                                widlimit=wl,
                                fixed=fixed, p_fix=p_fixed)
        else:
            model.do_mcmc(nwalkers=nwalkers, nchain=nchain, nburn=nburn,
                                fchain=chain_url,
                                fburn =burn_url,
                                set_verbose=verbose,
                                laglimit=ll,
                                widlimit=wl,
                                fixed=fixed, p_fix=p_fixed)




def MCMCrun_batch(targfolder,  mode = 'twoline', MCMC_params = MCMC_grade(), verbose=False, output_burn=False, output_cont=False, output_logp=False, skip_existing=True, skipexisting=True):
    '''
    MCMCrun_batch(targfolder, MCMC_params = MCMC_grade()

    Runs MCMCrun_single for every sim / subsamling in a batch

    Inputs

    Addtional Arguments
        Variable            Type        Default
        mode                str         twoline         What type of RM to do
                                                           -twoline (default) - Do both lines simultaneously
                                                           -line1  Do RM for only line 1
                                                            -line2  Do RM for only line 2
                                                            -Both   Do RM for both lines seperately
                                                            -all    Do both lines and also the twoline
        MCMC_params         MCMC_grade  MCMC_grade()    What MCMC parameters to run. Defaults to MCMC_Grade() default object
        verbose             bool        False           If True, outputs additional progress statements
        output_continuum    bool        False           If True and MCMC_params.do_continuum==True, will output the continuum MCMC chain file
        output_burn         bool        False           If True, will output burn steps in the MCMC chain to its own file
        output_logp         bool        False           If True, will also output the log likelihood of the main chain to a file
        skip_existing       bool        False           If True, will not run MCMC for existing files. NOT WORKING CURRENTLY
    '''

    #Read directory dir file to get folder urls
    nosims, no_subsamples, targdirs = read_dir(targfolder + "/_dir.txt")
    with open(targfolder+"/MCMC_params-%s.txt" %MCMC_params.name,'w') as f:
        f.write(MCMC_params.write_desc())
        f.close()
    np.save(arr=[MCMC_params], file = targfolder+"/MCMC_params-%s" %MCMC_params.name)

    #If some parameters are fixed, look for truth values in the param table
    if sum(MCMC_params.get_fixed_array())==0:
        #If no fixed values, don't worry about this
        pass
    else:
        param_url = targfolder + "/sim_params.npy"
        try:
            truth_value_params = np.load(param_url, allow_pickle=True)[0]
            if verbose: print("Loading fixed parameters from %s" %param_url)
        except:
            truth_value_params = sim_params()
            if verbose: print("Unable to locate sim params for truth values. Loading default")

    if verbose: print("Doing MCMC chains for all sims in %s" %targfolder)
    if verbose: print("Mode:\t%s" %mode)
    #For each folder from _dir.txt
    for i in range(nosims):
        if verbose: print("\t Doing MCMC for simulation %i" %i)

        for j in range(no_subsamples):
            folder = targdirs[i][j]
            print("\t\t Doing MCMC for folder %s" %folder)
            MCMCrun_single(targdir=folder,  mode = mode, MCMC_params=MCMC_params, verbose=verbose , truth_value_params = truth_value_params, output_burn=output_burn, output_cont=output_cont, output_logp=output_logp, skipexisting=skipexisting)
            if verbose: print("---")
        if verbose: print("======\n\n")

if __name__=="__main__":
    MCMCrun_batch(targfolder = "./Data/fakedata/Batch_Test", MCMC_params=runtest, verbose=True,mode='line1')