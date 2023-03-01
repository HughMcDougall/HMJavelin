'''
JaxTest.py


A test of basic jax / numpyro fitting
'''
#=========================================
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist, infer

import arviz as az
import corner

#from numpyro_ext.distributions import MixtureGeneral

#=========================================
#Generate Data
#=========================================

mixture_model = False
no_points = 15
outlier_fraction= 0.2
lin_params = [1,0] #m, c
outlier_error= 1

np.random.seed(5)

#Generate true data
x = np.sort( np.random.uniform(-2,2,no_points))
yerr = 0.2* np.ones_like(x)
y = lin_params[0] * x + lin_params[1] + yerr * np.random.randn(len(x))

#Generate outliers
m_bkg = np.random.rand(len(x)) < outlier_fraction
y[m_bkg] = np.sqrt(yerr[m_bkg]**2 + outlier_error**2) * np.random.randn(sum(m_bkg))

#Plot
plt.figure(1)
plt.errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
plt.scatter(x[m_bkg], y[m_bkg], marker="s", s=22, c="w", edgecolor="k", zorder=1000)
plt.scatter(x[~m_bkg], y[~m_bkg], marker="o", s=22, c="k", zorder=1000, label="data") #Note ~ inverts bools in m_bkg
#=========================================
#Define Model
#=========================================
numpyro.set_host_device_count(2)

def linear_model(x, yerr, y=None):
    '''
    The model we feed into numpyro
    '''
    #Define initial distributions for model parameters, namely angle and perpendicular offset
    theta = numpyro.sample("theta", dist.Uniform(-0.5 * jnp.pi, 0.5 * jnp.pi))
    b_perp = numpyro.sample("b_perp", dist.Normal(0,1))

    #Can _also_ track reparamaterizations of these same variables
    m = numpyro.deterministic("m", jnp.tan(theta))          #Lin Slope
    c = numpyro.deterministic("b", b_perp / jnp.cos(theta)) #Lin Offset / y intercept

    #Now, define likelihood function. This is included in the 'model' function
    #Here using a plate to contain the data
    with numpyro.plate("data",len(x)): #Make a plate called 'data' big enough for the data
        numpyro.sample("y", dist.Normal(m * x +c, yerr), obs=y) #Compare data to model

    '''
    Each of the 'sample' variables is defined with some distribution to get its likelihood from
    e.g. the parameters have their init distributions, the data has its comparison to the model
    collectively, everything contained within the function (later fed to NUTS) is used to get
    the likelihood for the MCMC sampling
    '''

#=========================================
#Alternative Mixture Model
#=========================================
def linear_mixture_model(x, yerr, y=None):
    
    #Use same variables (samples) for main linear fit:
    theta = numpyro.sample("theta", dist.Uniform(-0.5 * jnp.pi, 0.5 * jnp.pi))
    b_perp = numpyro.sample("b_perp", dist.Normal(0,1))
    m = numpyro.deterministic("m", jnp.tan(theta))          #Lin Slope
    b = numpyro.deterministic("b", b_perp / jnp.cos(theta)) #Lin Offset / y intercept
    fg_dist = dist.Normal(m*x+b,yerr) #Except define the foreground dist (main sequence) here

    #Define also the "background dist" variables
    bg_mean = numpyro.sample("bg_mean",dist.Normal(0,1))
    bg_sigma = numpyro.sample("bg_sigma",dist.HalfNormal(3))
    bg_dist = dist.Normal(bg_mean, jnp.sqrt(bg_sigma**2 + yerr**2))

    #Define also the outlier probability
    Q=numpyro.sample("Q",dist.Uniform(0,1))

    #Use a prebuilt gaussian mixture model
    mixture = MixtureGeneral(
        dist.Categorical(   probs = jnp.array([Q,1-Q]) ),    [fg_dist,bg_dist]
        )

    #Now, define likelihood function. This is included in the 'model' function
    #Here using a plate to contain the data
    with numpyro.plate("data",len(x)):
        y_ = numpyro.sample("obs", mixture, obs=y)

        log_probs = mixture.component_log_probs(y_)
        numpyro.deterministic( "p", log_probs- jax.nn.logsumexp(log_probs, axis = -1, keepdims=True))


#=========================================
#Do Fitting
#=========================================
#Now create a sampler to intake our data

if mixture_model==False:
    sampler = infer.MCMC(
        infer.NUTS(linear_model),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )

else:
    sampler = infer.MCMC(
        infer.NUTS(linear_mixture_model),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )
    
sampler.run(jax.random.PRNGKey(0), x, yerr, y=y)

#=========================================
#Visualize Data
#=========================================
inf_data = az.from_numpyro(sampler)
print(az.summary(inf_data))
corner.corner(inf_data, var_names=["m", "b"], truths=lin_params)

#=========================================
#Generate Fake Data from MCMC sampling
#=========================================
post_pred_samples = numpyro.infer.Predictive(linear_model,sampler.get_samples())
post_pred_samples = post_pred_samples(jax.random.PRNGKey(0), x, yerr)
post_pred_y       = post_pred_samples["y"]

plt.figure(1)
for y_fake in post_pred_y[np.random.randint(low=0,high=len(post_pred_samples["y"][0]),size=100)]:
    plt.plot(x, y_fake,  ".", color="C0", alpha=0.1)
    
plt.show()
