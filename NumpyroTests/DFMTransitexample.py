import warnings
from functools import partial

import corner
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import matplotlib as mpl
from exo4jax.light_curves import QuadLightCurve
from exo4jax.orbits import TransitOrbit
from numpyro import distributions as dist
from numpyro import infer
from numpyro_ext import distributions as distx
from numpyro_ext import info, optim
from tinygp import GaussianProcess, kernels
import arviz as az

from paths import figures



warnings.filterwarnings("ignore", category=FutureWarning)
jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(2)


def light_curve(params, t, period=1.0):
    lc = QuadLightCurve.init(u1=params["u"][0], u2=params["u"][1])
    orbit = TransitOrbit.init(
        period=period,
        duration=jnp.exp(params["log_duration"]),
        time_transit=params["t0"],
        impact_param=params["b"],
        radius=jnp.exp(params["log_r"]),
    )
    return jnp.exp(params["log_f0"]) * (1 + lc.light_curve(orbit, t)[0])


def build_gp(params, t, diag):
    kernel = jnp.exp(2 * params["log_amp"]) * kernels.quasisep.Matern32(
        jnp.exp(params["log_ell"])
    )
    return GaussianProcess(kernel, t, diag=diag, mean=partial(light_curve, params))


t_grid  = jnp.linspace(-0.3, 0.3, 1000)
t       = jnp.linspace(-0.2, 0.2, 75)
y_err   = 0.001

true_params = {
    "log_f0": 0.0,
    "u": jnp.array([0.3, 0.2]),
    "log_duration": jnp.log(0.12),
    "t0": 0.0,
    "b": 0.1,
    "log_r": jnp.log(0.1),
    "log_amp": jnp.log(0.002),
    "log_ell": jnp.log(0.02),
}

gp = build_gp(true_params, t, diag=y_err**2)
y = gp.sample(jax.random.PRNGKey(1047))
plt.plot(t, y, ".")
plt.xlabel("time")
plt.ylabel("relative flux");

def model(t, y_err, y=None, use_gp=True):
    # If we wanted to fit for all the parameters, we could use the following,
    # but we'll keep these fixed for simplicity.
    # log_duration = numpyro.sample("log_duration", dist.Uniform(jnp.log(0.08), jnp.log(0.2)))
    # b = numpyro.sample("b", dist.Uniform(0.0, 1.0))

    log_f0 = numpyro.sample("log_f0", dist.Normal(0.0, 0.01))
    u = numpyro.sample("u", distx.QuadLDParams())
    t0 = numpyro.sample("t0", dist.Normal(0.0, 0.1))
    log_r = numpyro.sample("log_r", dist.Normal(jnp.log(0.1), 2.0))
    numpyro.deterministic("r", jnp.exp(log_r))
    params = {
        "log_f0": log_f0,
        "u": u,
        "log_duration": jnp.log(0.12),
        "t0": t0,
        "b": 0.1,
        "log_r": log_r,
    }

    if use_gp:
        params["log_amp"] = numpyro.sample("log_amp", dist.Normal(0.0, 2.0))
        params["log_ell"] = numpyro.sample("log_ell", dist.Normal(0.0, 2.0))
        gp = build_gp(params, t, diag=y_err**2)
        numpyro.sample("y", gp.numpyro_dist(), obs=y)
        mu = gp.mean_function(t_grid)
        numpyro.deterministic("mu", mu)
        numpyro.deterministic("gp", gp.condition(y, t_grid, include_mean=False).gp.loc)

    else:
        numpyro.sample("y", dist.Normal(light_curve(params, t), y_err), obs=y)
        numpyro.deterministic("mu", light_curve(params, t_grid))


sampler_wn = infer.MCMC(
    infer.NUTS(
        model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=infer.init_to_value(values=true_params),
    ),
    num_warmup=1000,
    num_samples=2000,
    num_chains=2,
    progress_bar=True,
)
%time sampler_wn.run(jax.random.PRNGKey(11), t, y_err, y, use_gp=False)

inf_data_wn = az.from_numpyro(sampler_wn)
az.summary(inf_data_wn, var_names=["t0", "r"])

sampler = infer.MCMC(
    infer.NUTS(
        model,
        dense_mass=True,
        target_accept_prob=0.9,
        init_strategy=infer.init_to_value(values=true_params),
    ),
    num_warmup=1000,
    num_samples=2000,
    num_chains=2,
    progress_bar=True,
)
%time sampler.run(jax.random.PRNGKey(12), t, y_err, y, use_gp=True)

inf_data = az.from_numpyro(sampler)
az.summary(inf_data, var_names=["t0", "r"])

with mpl.rc_context({"font.size": 14}):
    p1 = inf_data.posterior
    p2 = inf_data_wn.posterior
    ranges = [
        (
            min(p1["t0"].values.min(), p2["t0"].values.min()),
            max(p1["t0"].values.max(), p2["t0"].values.max()),
        ),
        (
            0.07,
            # min(p1["r"].values.min(), p2["r"].values.min()),
            max(p1["r"].values.max(), p2["r"].values.max()),
        ),
    ]
    fig = corner.corner(
        inf_data_wn, range=ranges, bins=30, var_names=["t0", "r"], color="C1"
    )
    fig = corner.corner(
        inf_data,
        range=ranges,
        bins=30,
        var_names=["t0", "r"],
        labels=["$T_0$ [days]", r"$R_\mathrm{P} / R_\star$"],
        truths=[true_params["t0"], jnp.exp(true_params["log_r"])],
        color="C0",
        truth_color="k",
        fig=fig,
    )
    fig.savefig(figures / "transit_posteriors.pdf", bbox_inches="tight")

samples = sampler.get_samples()
pred_gp = samples["gp"] + samples["mu"]
pred = samples["mu"]

samples_wn = sampler_wn.get_samples()
pred_wn = samples_wn["mu"]

inds = jax.random.randint(jax.random.PRNGKey(0), (12,), 0, len(pred_gp))

plt.figure(figsize=(5, 3.5))
plt.errorbar(t, y, yerr=y_err, fmt=".k")
plt.plot(t_grid, pred_wn[inds].T, "C1", alpha=0.1)
plt.plot(t_grid, jnp.median(pred_wn, axis=0), "C1", label="without GP")
plt.plot(t_grid, pred_gp[inds].T, "C0", alpha=0.1)
plt.plot(t_grid, jnp.median(pred_gp, axis=0), "C0", label="with GP")
plt.plot(t_grid, light_curve(true_params, t_grid), "k--", lw=1, label="ground truth")
plt.xlim(-0.22, 0.22)
plt.legend(fontsize=10, loc="lower right")
plt.xlabel("time [days]")
plt.ylabel("relative flux")
plt.savefig(figures / "transit.pdf", bbox_inches="tight")
