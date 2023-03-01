import matplotlib.pyplot as plt

plt.plot()  # Required to reset the rcParams for some reason

import warnings
from functools import partial
import os

import arviz as az
import corner
import jax
import jax.numpy as jnp
import jaxopt
import numpyro
import tinygp
import matplotlib as mpl
from astropy.table import Table
from numpyro import distributions as dist
from numpyro import infer
from tinygp import GaussianProcess, kernels, transforms

warnings.filterwarnings("ignore", category=FutureWarning)
numpyro.set_host_device_count(2)

data = Table.read("araa-gps-main/src/data/quasar.csv")

#==========================================================================

@tinygp.helpers.dataclass
class Multiband(kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])

def time_delay_transform(lag, X):
    t, band = X
    try:
        return t - lag[band-1]
    except:
        return t - lag

def mean_func(means, X):
    t, band = X
    return means[band]

#==========================================================================

#DATA EXTRACTION
multiline_test = True
if False:
    scaletest=1
    N = len(data)
    X = jnp.concatenate((data["jd"].value, data["jd"].value)), jnp.concatenate(
        (jnp.zeros(N, dtype=int), jnp.ones(N, dtype=int))
    )
    y = jnp.concatenate((data["a_mag"].value, data["b_mag"].value*scaletest))
    diag = jnp.concatenate((data["a_mag_err"].value, data["b_mag_err"].value*scaletest)) ** 2
else:
    #Read OzDes File
    #Currently only reads line 1. To adapt to line 2 momentarily
    os.chdir('..')
    from reftable import *
    os.chdir('./NumpyroTests')

    homeurl="../Data/fakedata/FullBatch_0920/sim-01/00-OD_2linegood"

    Tcont, Xcont, Econt = unpack_source(homeurl+"/cont.dat")
    Tline1, Xline1, Eline1 = unpack_source(homeurl+"/line1.dat")
    Tline2, Xline2, Eline2 = unpack_source(homeurl+"/line2.dat")

    #VERY MESSY EXAMPLE DO NOT LOOK
    if not multiline_test:
        #Oneline
        X       = jnp.concatenate(  [ Tcont,                            Tline1]    )
        indices = jnp.concatenate(  [ jnp.zeros(len(Tcont),dtype=int),  jnp.ones(len(Tline1),dtype=int)]    )
        y       = jnp.concatenate(  [ Xcont,                            Xline1])
        diag = jnp.concatenate(     [ Econt,                            Eline1])
    else:
        #twoline
        X       = jnp.concatenate(  [ Tcont,                            Tline1,                             Tline2]    )
        indices = jnp.concatenate(  [ jnp.zeros(len(Tcont),dtype=int),  jnp.ones(len(Tline1),dtype=int),    jnp.ones(len(Tline2),dtype=int)+1]    )
        y       = jnp.concatenate(  [ Xcont,                            Xline1,                             Xline2])
        diag = jnp.concatenate(     [ Econt,                            Eline1,                             Eline2])
    X=(X,indices)


def build_gp(params, X, diag):
    # Do the sorting _before_ building the GP
    band = X[1]
    t = time_delay_transform(params["lag"], X)
    inds = jnp.argsort(t)

    kernel = Multiband(
        amplitudes=params["amps"],
        kernel=kernels.quasisep.Matern32(jnp.exp(params["log_ell"])),
    )

    mean = partial(mean_func, params["means"]) #Format 'mean_func' to require only an X input, so it works as a in-out function. Necessary to feed to Gaussian{Process(). Already defined it in a way that makes "X" input a time / index tuple, and formatted kernel (multiband) to read that sort of input appropriately
    return (
        GaussianProcess(kernel, (t[inds], band[inds]), diag=diag[inds], mean=mean),
        inds,
    )

#@jax.jit
def loss(params):
    gp, inds = build_gp(params, X, diag)
    return -gp.log_probability(y[inds])

#==========================================================================
if not multiline_test:
    true_params = {
        "lag": 420.0,
        "log_ell": jnp.log(100.0),
        "amps": jnp.array([0.08, 0.12*2]),
        "means": jnp.array([17.43, 17.53]),
    }
else:
    true_params = {
        "lag": jnp.array([420, 400]),
        "log_ell": jnp.log(100.0),
        "amps": jnp.array([0.08, 0.12*2, 0.12]),
        "means": jnp.array([17.43, 17.53,17.53]),
    }


plt.figure()
gp, inds = build_gp(true_params, X, diag)

#Make some fact data that emulate the simulation using known data. Basically the same as our mockup mirroring procedure
y = jnp.empty_like(y)
y = y.at[inds].set(gp.sample(jax.random.PRNGKey(10)))

plt.plot(X[0][X[1] == 0], y[X[1] == 0], ".", label="a")
plt.plot(X[0][X[1] == 1], y[X[1] == 1], ".", label="b")

print("Mock Data Generated")

opt = jaxopt.ScipyMinimize(fun=loss)

init = dict(true_params)
minimum = loss(init), init
lags = []
vals = []

print("Starting lag search")
for lag in jnp.linspace(0, 1000, 32): #Decreased count to 32
    if not multiline_test:
        init["lag"] = lag
    else:
        init["lag"] = jnp.array([lag]*int(max(X[1])))

    soln = opt.run(init)
    lags.append(soln.params["lag"])
    vals.append(soln.state.fun_val)

    if soln.state.fun_val < minimum[0]:
        minimum = soln.state.fun_val, soln.params
print("Gridsearch Done")
init = minimum[1]

plt.figure()
plt.plot(lags, vals, ".", alpha=0.2)
plt.xlabel("lag [days]")
plt.ylabel("loss minimized over other parameters");

#Time Domain for Plotting
t_lagged = X[0] - max(minimum[1]["lag"]) * X[1]
t_grid = jnp.linspace(t_lagged.min() - 200, t_lagged.max() + 200, 1000)

#==========================================================================

def model(X, diag, y):
    lag = numpyro.sample("lag", dist.Uniform(0.0, 1000.0))  #Delay
    log_ell = numpyro.sample("log_ell", dist.Uniform(jnp.log(10), jnp.log(1000.0))) #Lengthscale
    amps = numpyro.sample("amps", dist.Uniform(-5.0, 5.0), sample_shape=(2,)) #Band scaling
    mean_a = numpyro.sample("mean_a", dist.Uniform(17.0, 18.0)) #Average of band 0
    delta_mean = numpyro.sample("delta_mean", dist.Uniform(-2.0, 2.0))  #Diff in averages. Odd param but I guess easier to set prior
    means = jnp.stack((mean_a, mean_a + delta_mean))

    #Wrapper to more easily talk to 'build GP', which takes a dictionary in the way we formatted it here
    params = {
        "lag": lag,
        "log_ell": log_ell,
        "amps": amps,
        "means": means,
    }

    gp, inds = build_gp(params, X, diag)
    numpyro.sample("y", gp.numpyro_dist(), obs=y[inds])

    #Condition the GP with each band individually. Note that the X vals are the x vals of the output, not input.
    numpyro.deterministic(
        "pred_a",
        gp.condition(y[inds], (t_grid, jnp.zeros_like(t_grid, dtype=int))).gp.loc,
    )
    numpyro.deterministic(
        "pred_b",
        gp.condition(y[inds], (t_grid, jnp.ones_like(t_grid, dtype=int))).gp.loc,
    )

    #Turning this on breaks the chain. Unclear why
    if multiline_test:
        numpyro.deterministic(
            "pred_c",
            gp.condition(y[inds], (t_grid, jnp.ones_like(t_grid, dtype=int)+1)).gp.loc,
        )


init_params = dict(minimum[1])
init_params["mean_a"] = init_params["means"][0]
init_params["delta_mean"] = init_params["means"][1] - init_params["means"][0]

print("Beginning sampling")
sampler = infer.MCMC(
    infer.NUTS(
        model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=infer.init_to_value(values=init_params), #NumPyro model can take dictionary of parameters if names line up
    ),
    num_warmup=100,
    num_samples=200,
    num_chains=20,
    progress_bar=True,
)
sampler.run(jax.random.PRNGKey(12), X, diag, y)

print(" CHAIN DONE ")

#==========================================================================

inf_data = az.from_numpyro(sampler)
az.summary(inf_data, var_names=["lag", "delta_mean"])

if not multiline_test:
    with mpl.rc_context({"font.size": 14}):
        fig = corner.corner(
            inf_data,
            var_names=["lag", "delta_mean"],
            labels=["time delay [days]", "mean magnitude offset"],
            truths=[true_params["lag"], jnp.diff(true_params["means"])[0]],
        )
else:
    with mpl.rc_context({"font.size": 14}):
        fig = corner.corner(
            inf_data,
            var_names=["lag"],
            labels=["lag", "lag2"],
            truths=true_params["lag"],
        )

#==========================================================================
#PLOTTING PHASE
samples = sampler.get_samples()
lag = jnp.median(samples["lag"])
pred_a = samples["pred_a"]  #Samples indexed by name, including deterministic
pred_b = samples["pred_b"]
try:
    pred_c = samples["pred_c"]
except:
    pass

plotinds = jax.random.randint(jax.random.PRNGKey(134), (12,), 0, len(pred_a)) #Get indices of random samples in the chain

offset = 0.3

plt.figure(figsize=(5, 3.5))
#Grabs random samples from pred_a, which ranges over random sets of params. pred_a is conditioned at each sample in the MCMC
plt.plot(t_grid + lag, pred_a[plotinds, :].T, c="C0", alpha=0.3, lw=0.5)
plt.plot(t_grid + lag, pred_b[plotinds, :].T + offset, c="C1", alpha=0.3, lw=0.5)
try:
    plt.plot(t_grid + lag, pred_c[plotinds, :].T + offset, c="C1", alpha=0.3, lw=0.5)
except:
    pass

a = X[1] == 0 #boolean switch True such that band index == 0
plt.errorbar(
    X[0][a] + lag, #Moves cont forward instead of respone back to apply lag
    y[a],
    yerr=jnp.sqrt(diag[a]), #Just plotting input data /w errors. Remember 'diag' is 'measurement uncertainty in DFM work'
    fmt="oC0",
    label="A",
    markersize=4,
    linewidth=1,
)
b = X[1] == 1
plt.errorbar(
    X[0][b],
    y[b] + offset,
    yerr=jnp.sqrt(diag[b]),
    fmt="oC1",
    label="B",
    markerfacecolor="white",
    markersize=4,
    linewidth=1,
)
if multiline_test:
    c = X[1] == 2
    plt.errorbar(
        X[0][c],
        y[c] + offset * 2,
        yerr=jnp.sqrt(diag[b]),
        fmt="oC1",
        label="C",
        markerfacecolor="white",
        markersize=4,
        linewidth=1,
    )

plt.ylim(plt.ylim()[::-1])
plt.xlabel(f"time [days; A + {lag:.0f} days]")
plt.ylabel(f"magnitude [B + {offset}]")
plt.xlim(t_grid.min() + lag, t_grid.max() + lag)
plt.legend(loc="lower right", fontsize=10)

plt.show()