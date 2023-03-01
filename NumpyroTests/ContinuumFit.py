'''
Help

HM - 22/2
'''
import matplotlib.pyplot as plt

import warnings
import os
from functools import partial

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

#============================================

#Settup
plt.plot()  # Required to reset the rcParams for some reason
plt.close()

warnings.filterwarnings("ignore", category=FutureWarning)
numpyro.set_host_device_count(1)

os.chdir('..')
from reftable import *
os.chdir('./NumpyroTests')

#============================================

#Build GP for tiny GP
def build_gp(params, X, E):
    sigma_c, tau_d, mean = params

    base_kernel = kernels.quasisep.Matern32(scale=sigma_c, sigma=tau_d)
    meanfunc = lambda x: mean

    out = GaussianProcess(base_kernel, X, diag=E, mean=meanfunc)
    return(out)


#build numpyro Model
def model(X,Y,E):
    scale_range = [jnp.log10(jnp.std(Y))-1,    jnp.log10(jnp.std(Y))+1]
    scale_range = [-4,12]
    #scale_range = [0, 12]


    log_sigma  = numpyro.sample('log_sigma',numpyro.distributions.Uniform(scale_range[0],scale_range[1]))
    log_tau       = numpyro.sample('log_tau',numpyro.distributions.Uniform(-10,10))
    cont_mean       =numpyro.sample('mean',numpyro.distributions.Uniform(jnp.mean(Y)-3*jnp.std(Y),jnp.mean(Y)+3*jnp.std(Y)))

    cont_scale      = numpyro.deterministic('cont_scale',   jnp.power(10,log_sigma))
    tau_d           = numpyro.deterministic('tau_d',        jnp.power(10,log_tau))

    #Use build_gp() to build a tinygp gaussian process object
    params_to_gp = [cont_scale, tau_d, cont_mean]
    gp = build_gp( params_to_gp, X, E)

    #Sample using the numpyro friendly distribution that it has baked in
    numpyro.sample('y_cont', gp.numpyro_dist(), obs=Y)

    '''
    lag     =   numpyro.sample()
    scale   =   numpyro.sample()
    '''

#============================================
#Do Numpryo run
print("Creating sampler")

sampler = infer.MCMC(
    infer.NUTS(model),
    num_warmup=300,
    num_samples=400,
    num_chains=10,
    progress_bar=True,
)

#Get Data
X,Y,E = OD_2linegood.T_cont,    OD_2linegood.X_cont,    OD_2linegood.E_cont

sampler.run(jax.random.PRNGKey(1),X,Y,E)
print("sampling done")

#============================================
#Plotting
print("Beginning Plotting")

datachain = sampler.get_samples()
data=np.vstack([datachain['log_sigma'],datachain['log_tau'],datachain['mean']]).T

c = ChainConsumer()
c.add_chain(data, parameters=["$Lsig$", "$Ltau$", "$mean$"])
c.configure(summary=True, cloud=True, sigmas=np.linspace(0, 5, 20))
cfig=c.plotter.plot( truth = [2.38,-0.56,2.36])
cfig.tight_layout()

#numpyro.render_model(model,model_args=(X,Y,E))
plt.show()
