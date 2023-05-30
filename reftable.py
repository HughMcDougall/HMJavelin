import numpy as np
from warnings import warn

'''
reftable.py

Storage file for useful functions and data sources. Mostly acts as a holder for information used elsewhere.

HM 9/9

Changes
12/9 
- Changed longchain_test to use 2 walkers (must be even)
- Added check to mcmcparam object to make sure walkers are even

[MISSINGNO]
Changes from 13/9-18/9. Mostly sanity checks and parameter fixing utility

19/9
Added width limit to MCMC_params. Defaults to [0,60]
Added runtype modes to get_truth_values() and get_fixed()
'''


# ===============================
# UTILITY FUNCTIONS
# ===============================

def read_dir(url):
    '''
    nosims, nogrades, targdirs  = read_dir(url)

    Quick utility to read information from a directory '_dir' file.
    Placed here for convenience, as it is re-used often

    Arguments
        url         str         The url of the directory file

    Returns
        nosims      int         Number of simulations in the batch
        nogrades    int         Number of subsampling 'grades' for each simulation realization
        targdirs    [[str]]     a sorted list containing the urls of each output folder. Organised such that targdirs[i][j] is simulation 'i' and sampling 'j'
    '''
    dirfile = open(url)
    targdirs = []
    firstline = True
    for line in dirfile:
        if firstline:
            nosims = int(line.split('\t')[0])
            nogrades = int(line.split('\t')[1])
            firstline = False
        else:
            targdirs.append(line.removesuffix("\n"))
    dirfile.close()
    targdirs = [targdirs[sim * nogrades:(sim + 1) * nogrades] for sim in range(nosims)]

    return(nosims, nogrades, targdirs)
    # ===============================


def unpack_source(url):
    '''
    Tout,Xout,Eout = unpack_source(url)

    Loads a .dat file at 'url' and unpacks into 3 np arrays
    '''

    A=np.loadtxt(url)

    Tout=A[:,0]
    Xout=A[:,1]
    Eout=A[:,2]

    return(Tout,Xout,Eout)

def runtypes_from_mode(mode):
    runtypes=[]
    if mode =='all' or mode=='twoline':
        runtypes.append('twoline')
    if mode == 'all' or mode=='line1' or mode=='both':
        runtypes.append('line1')
    if mode == 'all' or mode=='line2' or mode=='both':
        runtypes.append('line2')

    return(runtypes)


# ===============================
# UTILITY CLASSES
# ===============================

class baseobj:
    '''
    Object base class with some safety features. Internal use only
    '''
    def __init__(self):
        self.fixed=False
        self.fixed=True

    def __setattr__(self,key,value):

        if hasattr(self,"fixed"):
            if self.fixed and not hasattr(self,key): raise TypeError("Attempted to alter non existant atrribute %s" %key)

        super().__setattr__(key,value)

# ===============================

class data_source(baseobj):
    '''
    Simple wrapper for light curves that area easier to read
    '''

    def __init__(self,cont_url,line1_url,line2_url,name=None):
        self.fixed= False

        self.cont_url   =   cont_url
        self.line1_url  =   line1_url
        self.line2_url  =   line2_url

        T_cont,X_cont,E_cont = unpack_source(cont_url)
        self.T_cont     =   T_cont
        self.X_cont     =   X_cont
        self.E_cont     =   E_cont

        T_line1,X_line1,E_line1 = unpack_source(line1_url)
        self.T_line1    =   T_line1
        self.X_line1    =   X_line1
        self.E_line1    =   E_line1

        T_line2, X_line2, E_line2 = unpack_source(line2_url)
        self.T_line2    =   T_line2
        self.X_line2    =   X_line2
        self.E_line2    =   E_line2

        self.name       =   name

        self.fixed=True

    def write_desc(self):
        out=''

        if self.name!=None:
            out+='Data Source:\t %s \n' %self.name
        else:
            out+='Unnamed Data Source \n'

        out += '\tContinuum data source: \n%s \t' %self.cont_url
        out += '\tLine 1 data source: \n%s \t' %self.line1_url
        out += '\tLine 2 data source: \n%s \n' %self.line2_url

        return(out)

class sim_params(baseobj):
    '''
    An object class that is used to hold properties of an AGN signal.

    '''
    def __init__(self, dt=0.1, tmax = 180*2*8, cont_tau=400, siginf_cont=1, delay1 = 180, width1= 30, amp1 = 1, delay2 = 250, width2 = 30, amp2 = 1, name = None):

        self.fixed=False

        #Continuum Properties
        self.dt             =   dt
        self.tmax           =   tmax
        self.cont_tau       =   cont_tau
        self.siginf_cont    =   siginf_cont

        #Line 1 Properties
        self.delay1         =   delay1
        self.width1         =   width1
        self.amp1           =   amp1

        #Line 2 Properties
        self.delay2         =   delay2
        self.width2         =   width2
        self.amp2           =   amp2

        self.name           =   name

        self.fixed=True

    def write_desc(self):
        out = ''
        out+=("SIM PROPERTIES \n \n")

        out+=("tau = %f  \n" % self.cont_tau)
        out+=("baseline = %f \n" % self.tmax)
        out+=("sim dt = %f \n" % self.dt)

        out+=("Line 1: \n")
        out+=("line 1 delay = %f \n" % self.delay1)
        out+=("line 1 width = %f \n" % self.width1)
        out+=("line 1 amplitude = %f \n" % self.amp1)

        out+=("Line 2: \n")
        out+=("line 2 delay = %f \n" % self.delay2)
        out+=("line 2 width = %f \n" % self.width2)
        out+=("line 2 amplitude = %f \n" % self.amp2)

        return(out)

    def get_truth(self,runtype=None):
        '''
        Returns javelin friendly truth values for use in fixed values of MCMC searches
        '''

        DEBUGLOG = False

        if DEBUGLOG:
            out = [self.cont_tau, self.siginf_cont, self.delay1, self.width1, self.amp1, self.delay2, self.width2, self.amp2]
        else:
            out = [np.log(self.siginf_cont), np.log(self.cont_tau), self.delay1, self.width1, self.amp1, self.delay2, self.width2, self.amp2]

        if runtype == 'line1':
            #If line 1, keep first to and middle 3 parameters
            out = out[:5]
        elif runtype == 'line2':
            #If line 2, keep first 2 and last 3 parameters
            out1 = out[:2]
            out1.extend(out[-3:])
            out=out1

        return(out)


class subsampling_grade(baseobj):
    '''
    Wrapper for subsampling parameters without a direct source to ape
    '''
    def __init__(self, cadence_cont = 30, E_cont=15/100, DE_cont = 5/100, cadence_line = 60, E_line = 500 / 100,  DE_line = 100 / 100,  tseason=180, name="unnamed_ss_grade"):
        self.fixed=False

        #Continuum Measurements
        self.cadence_cont   = cadence_cont
        self.E_cont         = E_cont
        self.DE_cont        = DE_cont

        # Line Measurements
        self.cadence_line   = cadence_line
        self.E_line         = E_line
        self.DE_line        = DE_line

        self.tseason        = tseason

        self.name           = name

        self.fixed=True

    def write_desc(self):
        out = ''
        #Continuum Measurements
        out+=   self.name+'\n'
        out+=   'Continuum Measurement Cadence = %s \n'   %self.cadence_cont
        out+=   'Continuum Measurement Average Error = %s \n'   %self.E_cont
        out+=   'Continuum Measurement STD of Error= %s \n'   %self.DE_cont

        # Line Measurements
        out+=   'Line Measurement Cadence = %s \n'   %self.cadence_line
        out+=   'Line Measurmeent Average Error = %s \n'   %self.E_line
        out+=   'Line Measurmeent STD of Error  = %s \n'   %self.DE_line

        out+=   'Length of measurement season = %s \n'   %self.tseason

        return(out)



class MCMC_grade(baseobj):
    '''
    Wrapper for MCMC chain parameters
    '''
    def __init__(self, nwalkers= 100, nchain = 300, nburn = 100, contwalkers = 100, contchain = 300, contburn =100, do_continuum =False, fixed_conttau = False, fixed_contvar = False, fixed_widths = True, fixed_delays=False, fixed_linevar=False, name=None):
        self.fixed = False
        #-----------------

        #Main MCMC Parameters
        self.nwalkers       = nwalkers
        self.nchain         = nchain
        self.nburn          = nburn

        #Continuum MCMC parameters
        self.do_continuum   = do_continuum
        self.contwalkers    = contwalkers
        self.contchain      = contchain
        self.contburn       = contburn

        self.fixed_conttau  = fixed_conttau
        self.fixed_contvar  = fixed_contvar

        self.fixed_widths   = fixed_widths
        self.fixed_delays   = fixed_delays
        self.fixed_linevar  = fixed_linevar

        #Misc
        self.name           = name

        self.laglimits      = [0, 250 * 3]
        self.widlimits      = [0, 30  * 2]

        self.p_fix          =[0, 6,  0, 30, 1,   0, 30, 1]

        #-----------------
        self.fixed = True
        
    def write_desc(self):
        out=''

        out+=("\n Initial Continuum Search Params \n")
        out+=("contwalkers = %i \n" %self.contwalkers)
        out+=("Contchain = %i \n" % self.contchain)
        out+=("Contburn = %i \n" % self.contburn)
    
        out+=("\n Lag MCMC Params \n")
        out+=("nwalkers = %i \n" %self.nwalkers)
        out+=("nchain = %i \n" % self.nchain)
        out+=("nburn = %i \n" % self.nburn)
        out+=("maxlag  = %f \n" % self.laglimits[-1])
        
        return(out)

    def get_fixed_array(self,runtype=None):
        '''
        Returns fixed and free parameters as a javelin friendly list
        '''

        out = [1,1, 1,1,1 ,1,1,1]
        if self.fixed_contvar:
            out[0] = 0
        if self.fixed_conttau:
            out[1] = 0
        if self.fixed_delays:
            out[2] = 0
            out[5] = 0
        if self.fixed_widths:
            out[3] = 0
            out[6] = 0
        if self.fixed_delays:
            out[4] = 0
            out[7] = 0

        if runtype == 'line1':
            #If line 1, keep first to and middle 3 parameters
            out = out[:5]
        elif runtype == 'line2':
            #If line 2, keep first 2 and last 3 parameters
            out1 = out[:2]
            out1.extend(out[-3:])
            out  = out1

        return(out)

    def __setattr__(self, key, value):
        if key=="nwalkers" or key=="contwalkers":
            assert value % 2 ==0, "Must have even number of walkers"
        if key=="nwalkers":
            try:
                if value >=sum(2*self.get_fixed_array()): warn("May have too few walkers for MCMC")
            except:
                pass
        if key=="contwalkers":
            assert value >=2, "Must have at least 2 walkers for continuum"

        super().__setattr__(key,value)

#==========================================
#CHAINCONSUMER STYLING
continuum_color = 'blue'
twoline_color   = 'purple'
line1_color     = 'green'
line2_color     = 'orange'
contour_sigmas  =   [1,2]


#==========================================
#DATA SOURCES
ClearSignal     = data_source(cont_url= "Data/RealData/00 ClearSignal/cont.dat",
                              line1_url="Data/RealData/00 ClearSignal/line1.dat",
                              line2_url="Data/RealData/00 ClearSignal/line2.dat",
                              name= 'ClearSignal')

JavelinExample  = data_source(cont_url= "Data/RealData/01 Javelin Example/continuum.dat",
                              line1_url="Data/RealData/01 Javelin Example/yelm.dat",
                              line2_url="Data/RealData/01 Javelin Example/zing.dat",
                              name='JavelinExample')

OD_1linegood    = data_source(cont_url= "Data/RealData/03 OzDes 1 Line/2925344542_gBand.dat",
                              line1_url="Data/RealData/03 OzDes 1 Line/2925344542_Hbeta_exp.dat",
                              line2_url="Data/RealData/03 OzDes 1 Line/2925344542_Hbeta_exp.dat",
                              name='OD_1linegood')

OD_2linegood    = data_source(cont_url= "Data/RealData/04 OzDes 2 Line Good/2940510474_gBand.txt",
                              line1_url="Data/RealData/04 OzDes 2 Line Good/2940510474_CIV_exp.txt",
                              line2_url="Data/RealData/04 OzDes 2 Line Good/2940510474_MgII.txt",
                              name='OD_2linegood')

OD_2linebad     = data_source(cont_url= "Data/RealData/05 OzDes 2 Line Bad/gband.dat",
                              line1_url="Data/RealData/05 OzDes 2 Line Bad/CIV.dat",
                              line2_url="Data/RealData/05 OzDes 2 Line Bad/MGII.dat",
                              name='OD_2linebad')

source_B1       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2971028700_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2971028700_Hbeta_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2971028700_MgII",
                              name='Hbetasource-B1-2971028700')

source_B2       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2925552152_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2925552152_Hbeta_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2925552152_MgII",
                              name='Hbetasource-B2-2925552152')

source_B3       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2971086054_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2971086054_Hbeta_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2971086054_MgII",
                              name='Hbetasource-B3-2971086054')

source_B4       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2971134055_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2971134055_Hbeta_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2971134055_MgII",
                              name='Hbetasource-B4-2971134055')

source_B5       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2970604169_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2970604169_Hbeta_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2970604169_MgII",
                              name='Hbetasource-B5-2970604169')

source_B6       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2925858108_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2925858108_Hbeta_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2925858108_MgII",
                              name='Hbetasource-B6-2925858108')

source_A1       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2940510474_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2940510474_CIV_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2940510474_MgII",
                              name='CIVsource-A1-2940510474')

source_A2       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2938498296_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2938498296_CIV_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2938498296_MgII",
                              name='CIVsource-A2-2938498296')

source_A3       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2939317867_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2939317867_CIV_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2939317867_MgII",
                              name='CIVsource-A3-2939317867')

source_A4       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2925718880_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2925718880_CIV_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2925718880_MgII",
                              name='CIVsource-A4-2925718880')

source_A5       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2939622630_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2939622630_CIV_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2939622630_MgII",
                              name='CIVsource-A5-2939622630')

source_A6       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2970791376_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2970791376_CIV_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2970791376_MgII",
                              name='CIVsource-A6-2970791376')

source_A7       = data_source(cont_url= "../../OzDESRM-main/Data/Y6+SV_Processed_LC/gband/2925523772_gBand.txt",
                              line1_url="../../OzDESRM-main/Data/Y6+SV_Processed_LC/Line/2925523772_CIV_exp.txt",
                              line2_url="../../OzDESRM-main/Data/MgII_LC/2925523772_MgII",
                              name='CIVsource-A7-2925523772')


#==========================================
#MCMCM Grades
longchain_test  = MCMC_grade(nwalkers= 2,       nchain = 700,   nburn = 1,      contwalkers = 100,  contchain = 300,    contburn =150,  name='longchain_test')
ultrahighqual   = MCMC_grade(nwalkers= 1000,    nchain = 700,   nburn = 1,      contwalkers = 100,  contchain = 300,    contburn =150,  name='ultrahighqual' )
highqual        = MCMC_grade(nwalkers= 300,     nchain = 600,   nburn = 200,    contwalkers = 100,  contchain = 300,    contburn =150,  name='highqual' )
highqual_longburn        = MCMC_grade(nwalkers= 300,     nchain = 600,   nburn = 2000,    contwalkers = 100,  contchain = 300,    contburn =150,  name='highqual_longburn' )
lowqual         = MCMC_grade(nwalkers= 50,      nchain = 50,    nburn = 10,     contwalkers = 50,   contchain = 50,     contburn =10,   name='lowqual' )
runtest         = MCMC_grade(nwalkers= 16,      nchain = 5,     nburn = 1,      contwalkers = 4,    contchain = 5,      contburn =1,    name='runtest' )
long_burn       = MCMC_grade(nwalkers= 100,     nchain = 3000,  nburn = 600,   contwalkers = 100,    contchain = 300,      contburn =150,    name='long_burn' )
one_chain         = MCMC_grade(nwalkers= 16,      nchain = 1000,     nburn = 1,      contwalkers = 4,    contchain = 1000,      contburn =1,    name='one_chain' )

free_all_longchain           = MCMC_grade(nwalkers= 20, nchain = 600, nburn = 1, contwalkers = 100, contchain = 300, contburn =150, do_continuum =False, fixed_conttau = False, fixed_contvar = False, fixed_widths = True, fixed_delays=False, fixed_linevar=False, name="free_all")
restrict_all_longchain       = MCMC_grade(nwalkers= 20, nchain = 600, nburn = 1, contwalkers = 100, contchain = 300, contburn =100, do_continuum =False, fixed_conttau = False, fixed_contvar = False, fixed_widths = True, fixed_delays=False, fixed_linevar=False, name="restrict_all")

#==========================================
#SUBSAMPLING GRADES
JavelinExample_Fixed    = subsampling_grade(cadence_cont = 8,   E_cont=4.3/100, DE_cont = 1.0/100,  cadence_line = 8,   E_line = 4.3 / 100,     DE_line = 1.0 / 100,    tseason=180, name='JavelinExample-FixedVars')
OD_1linegood_fixed      = subsampling_grade(cadence_cont = 8.5, E_cont=10/100,  DE_cont = 7/100,    cadence_line = 60,  E_line = 44 / 100,      DE_line = 13 / 100,     tseason=180, name='OD_1linegood-FixedVars')
OD_2linegood_fixed      = subsampling_grade(cadence_cont = 8,   E_cont=13/100,  DE_cont = 8/100,    cadence_line = 60,  E_line = 2.4 / 100,     DE_line = 12 / 100,     tseason=180, name='OD_2linegood-FixedVars')
OD_2linebad_fixed       = subsampling_grade(cadence_cont = 8.5, E_cont=15/100,  DE_cont = 5/100,    cadence_line = 60,  E_line = 500 / 100,     DE_line = 100 / 100,    tseason=180, name='OD_2linebad-FixedVars')
