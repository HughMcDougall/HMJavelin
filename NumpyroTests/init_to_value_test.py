import jax
import jax.numpy as jnp
import numpyro
from numpyro import infer
import numpy as np

def model(data):
    #Cont
    m = numpyro.sample('m',     numpyro.distributions.Uniform(-5,5))
    c = numpyro.sample('c',     numpyro.distributions.Uniform(-5,5))

    params={'m': m, 'c':c}
    print(params)

    numpyro.sample('y', numpyro.distributions.Normal(data['x']*m + c, data['e']), obs=data['y'])


params = {
    'm': 1,
    'c': 2
}

data = {
    'x': jnp.linspace(0,10,11),
    'y': jnp.linspace(0,10,11) *params['m'] +params['c'] + np.random.randn(11),
    'e': jnp.abs(jnp.array(np.random.randn(11)))
}

sampler = infer.MCMC(
    infer.NUTS(
        model,
        init_strategy=infer.init_to_value(values=params),
    ),

    num_warmup=5,
    num_samples=5,
    num_chains=1,
    progress_bar=True,
)

sampler.run(jax.random.PRNGKey(12), data)