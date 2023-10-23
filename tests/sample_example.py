"""
BW, 23.10.23
Sample from banana or funnel,
    - stepsize is adapted
    - funnel exhibits weird behavious

"""


import jax
import jax.numpy as jnp
from rmhmc.sampling import sample
from rmhmc.hmc import hmc
from problems import banana, funnel
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(42)
    num_chains = 4
    num_samples = 1000
    system = funnel(False, False)
    kinetic_state = system.kinetic_tune_init(2)
    q0 = jnp.array([0.3, 0.5])
    initial_q = jnp.array(num_chains * [q0])

    kernel = hmc(system=system, num_steps=20)
    sampler_carry = sample(
        kernel=kernel,
        rng_key=rng_key,
        initial_coords=initial_q,
        num_steps=num_samples,
        num_chains=num_chains,
    )
    # print(sampler_carry.state)
    states = sampler_carry.state
    print(states.q.shape)
    arr = states.q
    n_chains, n_samples, n_params = arr.shape
    arr_reshaped = arr.reshape(-1, n_params)
    print(arr_reshaped.shape)
    plt.scatter(arr_reshaped[:, 0], arr_reshaped[:, 1], alpha=0.3)
    plt.show()
