'''
Attempt at fitting N-band data

HM - 24/2
'''
import matplotlib.pyplot as plt

import warnings
import os

import arviz as az
import corner
from chainconsumer import ChainConsumer

import jax
import jax.numpy as jnp
import jaxopt
import numpyro
from numpyro import distributions as dist
from numpyro import infer
from tinygp import GaussianProcess, kernels, transforms
import tinygp

import matplotlib as mpl
from astropy.table import Table
from functools import partial



#============================================
#Settup

plt.plot()  # Required to reset the rcParams for some reason
plt.close()

warnings.filterwarnings("ignore", category=FutureWarning)
numpyro.set_host_device_count(1)

#Get
#PLACEHOLDER ONLY DO NOT LOOK
os.chdir('..')
from reftable import *
homeurl="./Data/fakedata/FullBatch_0920/sim-01/00-OD_2linegood"
Tcont, Xcont, Econt = unpack_source(homeurl + "/cont.dat")
Tline1, Xline1, Eline1 = unpack_source(homeurl + "/line1.dat")
Tline2, Xline2, Eline2 = unpack_source(homeurl + "/line2.dat")
os.chdir('./NumpyroTests')
#Assemble data
cont_data = {'T': Tcont, 'Y': Xcont, 'E': Econt}
line1_data = {'T': Tline1, 'Y': Xline1, 'E': Eline1}
line2_data = {'T': Tline2, 'Y': Xline2, 'E': Eline2}
#============================================
#Utility Funcs
def mean_func(means, X):
    '''
    Utitlity function to take array of constants and return as gp-friendly functions
    '''
    t, band = X
    return(means[band])

def LC_to_banded(lcs):
    '''
    Takes list of lightcurve objects and returns as single banded lightcurve
    '''
    Nbands = len(lcs)

    T = jnp.concatenate([lc['T'] for lc in lcs])
    Y = jnp.concatenate([lc['Y'] for lc in lcs])
    E = jnp.concatenate([lc['E'] for lc in lcs])

    bands = jnp.concatenate([jnp.zeros(len(lc['T']), dtype='int32') + band for band,lc in zip(range(Nbands),lcs)])

    out = {
        'T': T,
        'Y': Y,
        'E': E,
        'bands': bands,
    }

    return(out)

def banded_to_lc(data):
    '''
    Takes banded data and returns as list of individual data
    '''

    bands = data['bands']
    Nbands = jnp.max(bands)+1
    out=[]

    for i in range(Nbands):
        T = data['T'][bands==i]
        Y = data['Y'][bands==i]
        E = data['E'][bands==i]

        out.append({
            'T':T,
            'Y':Y,
            'E':E,
            })


    return(out)

#============================================
#Main Working Funcs

def make_mock_data(data, params, basekernel=tinygp.kernels.Exp):
    '''
    Takes banded LC data and params to fake
    :param data:
    :param params:
    :param basekernel:
    :return:
    '''

    #Make GP
    gp, inds = build_gp(data, params, basekernel)

    #Generate and fill mock data
    y_mock = jnp.empty_like(data['Y'])
    y_mock = y_mock.at[inds].set(gp.sample(jax.random.PRNGKey(10)))

    #Save results to copy of input data
    out = dict(data)
    out['Y'] = y_mock

    if 'means' in params.keys():
        y_mock+=params['means'][data['bands']]

    return(out)


def build_gp(data, params, basekernel=tinygp.kernels.Exp):
    '''
    Takes banded LC data and params, returns tinygp gaussian process
    :param data:
    :param params:
    :param basekernel:
    :return:
    '''

    #Unpack data and params
    T, Y, E= data['T'], data['Y'], data['E']

    tau = jnp.exp(params['log_tau'])
    sigma_c = jnp.exp(params['log_sigma_c'])

    if 'bands' in data.keys():
        bands = data['bands']
        cont_only  = False
        Nbands = jnp.max(bands)
    else:
        bands = jnp.zeros_like(T,dtype='int32')
        Nbands = 1
        cont_only = True


    means = params['means']

    if not cont_only:
        line_lags = params['lags']
        line_amps  = params['amps']

    #------------
    #Apply data tform
    #Offset, lag, scale
    Y /= jnp.where(bands == 0, sigma_c, 1) # Scale Continuum
    E /= jnp.where(bands == 0, sigma_c, 1)
    if not cont_only:
        T -= jnp.where(bands>0, line_lags[bands-1] , 0 ) #Apply Lags

        Y /= jnp.where(bands > 0, line_amps[bands - 1], 1) #Scale Line Signal & Errors
        E /= jnp.where(bands > 0, line_amps[bands - 1], 1)
    Y-=means[bands]

    #Sort data into gp friendly format
    sort_inds = jnp.argsort(T)

    #Make GP
    kernel = basekernel(scale = tau)
    gp = GaussianProcess(
        kernel,
        T[sort_inds],
        diag=E[sort_inds]**2,
                         )

    out = (gp, sort_inds)
    return(out)


def model(data):
    #Cont
    log_sigma   = numpyro.sample('log_sigma',   numpyro.distributions.Uniform(-5,5))
    log_tau     = numpyro.sample('log_tau',     numpyro.distributions.Uniform(0,10))

    Nbands = jnp.max(data['bands'])+1

    cont_scale  = numpyro.deterministic('cont_scale',   jnp.exp(log_sigma))
    tau_d       = numpyro.deterministic('tau_d',        jnp.exp(log_tau))

    #Lag and scaling of respone lines
    lags = numpyro.sample('lags', numpyro.distributions.Uniform(0,  180*4),  sample_shape=(Nbands-1,))
    amps = numpyro.sample('amps', numpyro.distributions.Uniform(0,  10),    sample_shape=(Nbands-1,))

    #Means
    means = numpyro.sample('means', numpyro.distributions.Uniform(-100,100), sample_shape=(Nbands,))

    params = {
        'log_tau': log_tau,
        'log_sigma_c': log_sigma,
        'lags': lags,
        'amps': amps,
        'means': means,
    }

    #build gp
    gp, sort_inds = build_gp(data, params)

    #Sample
    numpyro.sample('y', gp.numpyro_dist(), obs=data['Y'][sort_inds])

    #==========================================
    #Additional conditions. For later.
    if False:
        #Lag similarity prior
        numpyro.sample('lag similarity', numpyro.distributions.Normal(1,0.1), obs = lags[0] / lags[1])

        # Apply Addtl. Factors
        '''
        L = means[0]
        log10_RL_lag_1 = jnp.log10(L) * alpha[0] + beta[0]
        log10_RL_lag_2 = jnp.log10(L) * alpha[1] + beta[1]
    
        log10_lag1 = jnp.log10(lags[0])
        log10_lag2 = jnp.log10(lags[1])
    
        numpyro.factor('RL-line1', - ((log10_RL_lag_1 - log10_lag1) / delta[0] ) **2 - jnp.log(jnp.sqrt(2 * jnp.pi) * delta[0]))
        numpyro.factor('RL-line2' - ((log10_RL_lag_2 - log10_lag2) / delta[1] ) **2 - jnp.log(jnp.sqrt(2 * jnp.pi) * delta[1]))
        '''


#============================================
#Tests

#Test lc
real_data = LC_to_banded([cont_data, line1_data, line2_data])

true_params = {
    'log_tau': jnp.log(400),
    'log_sigma_c': 0,
    'lags': jnp.array([180, 250]),
    'amps': jnp.array([1, 1]),
    'means': jnp.array([0, 0, 0]),
}

#Test build gp
gp,inds = build_gp(real_data, true_params)

#Test mockup
mock_data = make_mock_data(real_data, true_params)
mock_lcs = banded_to_lc(mock_data)

#Test Plots
C = ['b', 'r', 'g']
N = len(mock_lcs)

fig,ax=plt.subplots(N,1)
for i in range(len(mock_lcs)):
    ax[i].errorbar(mock_lcs[i]['T'],mock_lcs[i]['Y'],mock_lcs[i]['E'], fmt='none', c=C[i])
    cent = jnp.median(mock_lcs[i]['Y'])
    wid = np.percentile(mock_lcs[1]['Y'],75) - np.percentile(mock_lcs[1]['Y'],25)
    ax[i].set_ylim(cent-2*wid, cent+2*wid)

plt.show()

#============================================
#Run
#Make sampler
init_params = dict(true_params)

print("Beginning sampling :)")
sampler = infer.MCMC(
    infer.NUTS(
        model,
        init_strategy=infer.init_to_value(values = init_params),
    ),

    num_warmup=1000,
    num_samples=6000,
    num_chains=1,
    progress_bar=True,
)

#run
sampler.run(jax.random.PRNGKey(12), mock_data)

c = ChainConsumer()
c.add_chain(sampler.get_samples()['lags'], parameters=["lag1", "lag2"])
c.configure(summary=True, cloud=True, sigmas=np.linspace(0, 2, 3))
cfig=c.plotter.plot( truth =np.array(true_params['lags']))
cfig.tight_layout()
plt.show()