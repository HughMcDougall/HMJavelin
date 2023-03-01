'''
An attempt at the bayesian linear regression /w numpyro from:
https://num.pyro.ai/en/latest/tutorials/bayesian_hierarchical_linear_regression.html

HM 17/12
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from jax import random

from sklearn.preprocessing import LabelEncoder

import chainconsumer as cs
import arviz as az


#Import working data
train = pd.read_csv(
    "https://gist.githubusercontent.com/ucals/"
    "2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/"
    "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/"
    "osic_pulmonary_fibrosis.csv"
)

#For convenience, use sklearn LabelEncoder to add index numbers for patient ID's
patient_encoder = LabelEncoder()
train["patient_code"] = patient_encoder.fit_transform(train["Patient"].values)

#=============================
def chart_patient(patient_ID, axes):
    '''
    Plots the data for patient_ID to target matplotlib axes
    '''

    #Extract data from the training data
    data = train[ train["Patient"]==patient_ID ]
    x=data["Weeks"]
    y=data["FVC"]

    axes.set_title(patient_ID)
    sns.regplot(x=x, y=y, ax=axes, ci=None, line_kws={"color": "red"})
#=============================

def model(patient_IDs, Weeks, FVC_obs = None):
    '''
    :param patient_ID:
    :param Weeks:
    :param FVC_obs:
    :return:
    '''

    #Define hyperparameters / hyperpriors
    μ_α = numpyro.sample("μ_α", dist.Normal(0.0, 500.0))
    σ_α = numpyro.sample("σ_α", dist.HalfNormal(100.0))
    μ_β = numpyro.sample("μ_β", dist.Normal(0.0, 3.0))
    σ_β = numpyro.sample("σ_β", dist.HalfNormal(3.0))

    #Find no. patients in our ID input
    n_patients = len(np.unique(patient_IDs))

    #Define a plate with the linear parameters
    with numpyro.plate("plate_i",n_patients):
        α = numpyro.sample("α", dist.Normal(μ_α, σ_α))
        β = numpyro.sample("β", dist.Normal(μ_β, σ_β))

    #Create also a variable for the spread of the model. This is out of plate / constant over all instances
    σ = numpyro.sample("σ", dist.HalfNormal(100.0))

    #Make linear model(s) with our params
    FVC_est = α[patient_IDs] + β[patient_IDs] * Weeks

    #Now generate a plate with all data sets (i.e. repeat patients)
    with numpyro.plate("data", len(patient_IDs)):
        numpyro.sample("obs", dist.Normal(FVC_est, σ), obs=FVC_obs)


#=============================
#Get the important data, use .values to get np arrays instead of pd dataframes
FVC_obs         = train["FVC"].values
Weeks           = train["Weeks"].values
patient_code    = train["patient_code"].values

#Make a NUTS kernel out of our linear model
nuts_kernel = NUTS(model)

#Make an MCMC instance
mcmc = MCMC(nuts_kernel, num_samples=2000, num_warmup=2000)
#Set the RNG key for reproducability
rng_key = random.PRNGKey(0)
#Fire away
mcmc.run(rng_key, patient_code, Weeks, FVC_obs=FVC_obs)

posterior_samples = mcmc.get_samples()

data = az.from_numpyro(mcmc)
az.plot_trace(data, compact=True, figsize=(15, 25))