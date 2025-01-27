from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax, random

from rmhmc.base_types import Array, Momentum
from rmhmc.hamiltonian import euclidean, riemannian

from problems import banana_logprob_and_metric

L = np.random.default_rng(9).normal(size=(5, 5))
L[np.diag_indices_from(L)] = np.exp(L[np.diag_indices_from(L)])
L[np.triu_indices_from(L, 1)] = 0.0


@pytest.mark.parametrize(
    "cov",
    [
        3.5 * jnp.eye(2),
        10.0 * jnp.eye(5),
        jnp.diag(np.random.default_rng(77).uniform(size=4)),
        L @ L.T + 1e-1 * jnp.eye(L.shape[0]),
    ],
)
def test_sample_momentum_euclidean(cov: Array) -> None:
    ndim = cov.shape[0]
    system = euclidean(lambda x: 0.5 * jnp.sum(x**2), cov=cov)
    kinetic_state = system.kinetic_tune_init(ndim)

    def _sample(
        key: random.KeyArray, _: int
    ) -> Tuple[random.KeyArray, Momentum]:
        key1, key2 = random.split(key)
        return key2, system.sample_momentum(
            kinetic_state, jnp.zeros(ndim), key1
        )

    _, result = lax.scan(_sample, random.PRNGKey(5), jnp.arange(100_000))

    np.testing.assert_allclose(
        jnp.dot(cov, jnp.cov(result, rowvar=0)), np.eye(ndim), atol=0.05
    )


@pytest.mark.parametrize(
    "q",
    [jnp.array([0.1, 0.3]), jnp.array([-0.5, -0.5]), jnp.array([0.0, 0.0])],
)
def test_sample_momentum_riemannian(q: Array) -> None:
    log_posterior, metric = banana_logprob_and_metric()
    system = riemannian(log_posterior, metric)
    kinetic_state = system.kinetic_tune_init(2)

    M = metric(q)

    def _sample(
        key: random.KeyArray, _: int
    ) -> Tuple[random.KeyArray, Momentum]:
        key1, key2 = random.split(key)
        return key2, system.sample_momentum(kinetic_state, q, key1)

    _, result = lax.scan(_sample, random.PRNGKey(5), jnp.arange(1_000_000))
    np.testing.assert_allclose(jnp.cov(result, rowvar=0), M, atol=0.05)
