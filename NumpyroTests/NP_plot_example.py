
import jax.numpy as jnp
import numpyro
from tinygp import GaussianProcess, kernels


def build_gp(params, X, E):
    sigma_c, tau_d, mean = params

    base_kernel = kernels.quasisep.Matern32(scale=sigma_c, sigma=tau_d)
    meanfunc = lambda x: mean

    out = GaussianProcess(base_kernel, X, diag=E, mean=meanfunc)
    return(out)

def model(X,Y,E):

    scale = numpyro.sample('scale', numpyro.distributions.Uniform(0,10))

    gp = GaussianProcess(kernels.quasisep.Matern32(scale=scale), X, diag=E)

    numpyro.sample('y_cont', gp.numpyro_dist(), obs=Y)


X = jnp.ones(10)
Y = jnp.ones(10)
E = jnp.ones(10)

graph = numpyro.render_model(model, model_args=(X,Y,E,), filename="example_autograph.pdf")

print("Done")

