'''
BH Linear Model.py

A personal test for using multivariable priors in heirarchical analysis

HM 23
'''

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist, infer

import arviz as az
import corner
from chainconsumer import ChainConsumer

#Generate Fake Data
no_sources = 100
min_data = 5
max_data = 30
xmin = -5
xmax = 5
errormag = 0.01

M_true = 0.5
C_true = 0.1
SIG_true = 0.1


#make true slopes
cs = np.random.rand(no_sources)
ms = M_true*cs  +   np.random.randn(no_sources)*SIG_true + C_true

#Make data for each slope
I = np.random.randint(low=min_data, high=max_data, size=no_sources)

X = [np.array(np.random.rand(i)*(xmax-xmin) + xmin) for i in I]
Y = [ms[j]*X[j] + cs[j] for j in range(no_sources)]
E = [(np.random.poisson(lam=1, size=I[j])+1) * errormag for j in range(no_sources)]


fig,ax = plt.subplots(1,2)
ax[0].scatter(cs,ms)
ax[0].axline([0,C_true] , slope=M_true , c='r', ls='-')
ax[0].axline([0,C_true+SIG_true*2] , slope=M_true , c='r', ls='--')
ax[0].axline([0,C_true-SIG_true*2] , slope=M_true , c='r', ls='--')
ax[0].title.set_text('Scatter of m and c')


for j in range(no_sources):
    ax[1].errorbar(X[j],Y[j],E[j],fmt='none')
    ax[1].axline(xy1=[0,cs[j]],slope=ms[j])
ax[1].title.set_text('All Dists')
plt.show()


#==========================================

def model_one(I,X,E=None,Y=None):
    
    #Population Properties
    C   = numpyro.sample("C", dist.Uniform(-1, 3))
    M   = numpyro.sample("M", dist.Uniform(-2,6) )

    SIG = numpyro.sample("SIG", dist.Uniform(0,2) )

    no_sources = len(np.unique(I))

    #Source Properties
    with numpyro.plate("sources",no_sources):
        #Vague priors for source
        c = numpyro.sample("c", dist.Uniform(-2, 6))
        m = numpyro.sample("m", dist.Uniform(-1, 3))
        pred_m = (M*c + C)
        numpyro.factor('linrel', -( ( (m-pred_m ) / SIG)**2))
        numpyro.factor('linrel_norm', - jnp.log(jnp.sqrt(2 * jnp.pi) * SIG))

        numpyro.sample('linrel', numpyro.distributions.Normal(0,SIG), obs=(m-pred_m) )

    #Fit data by source
    predicted = m[I]*X + c[I]
    with numpyro.plate("data", len(X)):
        numpyro.sample("obs", dist.Normal(predicted, E), obs=Y)

def model_two(I, X, E=None, Y=None):
    # Population Properties
    C = numpyro.sample("C", dist.Uniform(-1, 3))
    M = numpyro.sample("M", dist.Uniform(-2, 6))

    SIG = numpyro.sample("SIG", dist.Uniform(0, 2))

    no_sources = len(np.unique(I))

    with numpyro.plate("sources", no_sources):
        c = numpyro.sample("c", dist.Uniform(-2, 6))

        pred_m = (M * c + C)
        m = numpyro.sample("m", dist.Normal(pred_m, SIG))

    # Source Properties
    predicted = m[I] * X + c[I]

    with numpyro.plate("data", len(X)):
        numpyro.sample("obs", dist.Normal(predicted, E), obs=Y)


def model_three(I, X, E=None, Y=None):
    #Population Properties
    C   = numpyro.sample("C", dist.Uniform(-1, 3))
    M   = numpyro.sample("M", dist.Uniform(-2,6) )

    SIG = numpyro.sample("SIG", dist.Uniform(0,2) )

    no_sources = len(np.unique(I))

    #Source Properties
    with numpyro.plate("sources",no_sources):
        #Vague priors for source
        c = numpyro.sample("c", dist.Uniform(-2, 6))
        m = numpyro.sample("m", dist.Uniform(-1, 3))
        pred_m = (M*c + C)

        numpyro.sample('linrel', numpyro.distributions.Normal(0,SIG), obs=(m-pred_m) )

#==========================================

#Sort data into friendly format
def sort(X):
    I = np.concatenate([[j for i in range(len(X[j]))] for j in range(len(X))])
    outx=np.concatenate(X)
    return(I,outx)

Idata=sort(X)[0]
Xdata=sort(X)[1]
Edata=sort(E)[1]
Ydata=sort(Y)[1]

sampler = infer.MCMC(
        infer.SA(model_one),
        num_warmup=400,
        num_samples=500,
        num_chains=10,
        progress_bar=True,
    )

sampler.run(jax.random.PRNGKey(0), Idata, Xdata, E=Edata, Y=Ydata)
print("Chains Done")

#==========================================
#Plot Data
print("Organising samples")
samples = sampler.get_samples()
Cs_samp , Ms_samp=  np.concatenate(samples['c']), np.concatenate(samples['m'])

print("Population Scatter")

plt.figure()
plt.scatter(Cs_samp , Ms_samp , s = 0.1, alpha=0.5, label='Samples')
plt.xlabel("c")
plt.ylabel("m")

plt.axline([0,C_true] , slope=M_true , c='r', ls='-', label='True Population Dist')
plt.axline([0,C_true+SIG_true*2] , slope=M_true , c='r', ls='--')
plt.axline([0,C_true-SIG_true*2] , slope=M_true , c='r', ls='--')

plt.scatter(cs,ms,c='r', label='True Source Values')

C_rec = np.median(samples['C'])
M_rec = np.median(samples['M'])
SIG_rec = np.median(samples['SIG'])

plt.axline([0,C_rec] , slope=M_rec , c='b', ls='-', label='Recovered Population Dist')
plt.axline([0,C_rec+SIG_rec*2] , slope=M_rec , c='b', ls='--')
plt.axline([0,C_rec-SIG_rec*2] , slope=M_rec , c='b', ls='--')

plt.legend(loc='best')
plt.show()

print("Doing Corner Plots")

c = ChainConsumer()
c.add_chain(np.vstack([samples['C'], samples['M'], samples['SIG']]).T, parameters = ["M", "C", "SIGMA"])
c.configure(summary=True, cloud=True, sigmas=np.linspace(0, 2, 3))
cfig=c.plotter.plot( truth = [C_true,M_true,SIG_true])
cfig.tight_layout()
plt.show()

d = ChainConsumer()
d.add_chain(np.vstack([Cs_samp, Ms_samp]).T, parameters = ["C", "M"])
d.configure(summary=True, cloud=True, sigmas=np.linspace(0, 2, 3))
dfig=d.plotter.plot()
dfig.tight_layout()
plt.show()

print("All Finished")