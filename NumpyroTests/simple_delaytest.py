'''
A test to use numpyro to estimate a delay between two DRW's
'''

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist, infer

import arviz as az
import corner

from Data_sim_Utils import DRW_sim, season_sample, tophat_convolve

def covar_mat(T1,T2,dT,tau,sig1,sig2):
    '''
    :param T1:
    :param T2:
    :param dT:
    :return:
    '''

    N=len(T1) + len(T2)

    A11, B11 = jnp.meshgrid(T1,T1)
    A12, B12 = jnp.meshgrid(T1,T2)
    A22, B22 = jnp.meshgrid(T2,T2)

    K11 = sig1**2 * jnp.exp( -abs(A11-B11) / tau )
    K22 = sig2**2 *jnp.exp( -abs(A22-B22) / tau )
    K12 = sig1*sig2*jnp.exp( -abs(A12-B12-dT) / tau )

    K = jnp.vstack([
        jnp.hstack([K11,K12.T]),
        jnp.hstack([K12,K22])
    ])

    return(K)


print(":)")
#Make kernel function with npy
#Do mcmc
#Do some realizations

def model(T1,T2,Y1,Y2):

    logsig1 = numpyro.sample("log_signal_1_var", dist.LogNormal(0.0, 10.0))
    logsig2 = numpyro.sample("log_signal_2_var", dist.LogNormal(0.0, 10.0))

    logtau = numpyro.sample("log_timescale", dist.LogNormal(0.0, 10.0))
    dT     = numpyro.sample("delay",   dist.HalfNormal(3))

    tau    = numpyro.deterministic('timescale',jnp.exp(logtau))
    sig1   = numpyro.deterministic('sig1',jnp.exp(logsig1))
    sig2   = numpyro.deterministic('sig2',jnp.exp(logsig2))

    Y   =   jnp.hstack([Y1,Y2])
    K   =   covar_mat(T1,T2,dT,tau,sig1,sig2)

    numpyro.sample(
        "Y",
        dist.MultivariateNormal(covariance_matrix=K),
        obs=Y,
    )


#==================================

#True Params
dT = 200
tau = 300
sig1 = 1
sig2 = 1

#Generate Fake Data
tmax    = 365*4
seasons = 4*2

T1_True, Y1_True = DRW_sim(dt=0.1, tmax=tmax, tau=tau, siginf=sig1, x0=None, E0=0, method='square')
Y2_True          = tophat_convolve(T1_True, Y1_True, tau, siginf=sig1, method='square',delay=dT,amp=sig2,width=1, delay_from_center=True)

T1, Y1, E1 = season_sample(T1_True, Y1_True, T_season = 30, dt=30, garble=False, rand_space=True)
T2, Y2, E2 = season_sample(T1_True, Y2_True, T_season = 30, dt=30, garble=False, rand_space=True)
Y1/=np.std(Y1)
Y2/=np.std(Y2)

#=========================================
print("Creating sampler")
sampler = infer.MCMC(
    infer.NUTS(model),
    num_warmup=100,
    num_samples=100,
    num_chains=4,
    progress_bar=True,
)



print("Running sampling")
sampler.run(jax.random.PRNGKey(0), T1,T2,Y1,Y2)
print('Sampling done :)')

inf_data = az.from_numpyro(sampler)
print(az.summary(inf_data))


corner.corner(inf_data, var_names=["log_timescale", "log_signal_1_var"], truths=[np.log(tau),np.log(sig1)])
plt.show()

plt.figure()
plt.plot(T1,Y1)
plt.plot(T2,Y2)
plt.legend()

plt.show()