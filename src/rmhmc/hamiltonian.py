__all__ = ["euclidean", "riemannian"]

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax.flatten_util import ravel_pytree

from .base_types import (
    Array,
    KineticFunction,
    KineticState,
    Momentum,
    Position,
    PotentialFunction,
    Scalar,
)
from .integrator import (
    IntegratorInitFunction,
    IntegratorUpdateFunction,
    implicit_midpoint,
    leapfrog,
)


@dataclass(frozen=True)
class EuclideanKineticState(KineticState):
    count: Scalar
    tril: Array
    mu: Array
    m2: Array


class System(NamedTuple):
    # Functions to evaluate the potential and kinetic energies
    potential: PotentialFunction
    kinetic: KineticFunction

    # Function to sample the momentum at a specific state and position
    sample_momentum: Callable[
        [KineticState, Position, random.KeyArray], Momentum
    ]

    # Functions for tuning the kinetic energy function
    kinetic_tune_init: Callable[[int], KineticState]
    kinetic_tune_update: Callable[[KineticState, Position], KineticState]
    kinetic_tune_finish: Callable[[KineticState], KineticState]

    # System-specific integrator functions
    integrator_init: IntegratorInitFunction
    integrator_update: IntegratorUpdateFunction


def euclidean(
    log_probability_fn: Callable[[Position], Scalar], diagonal: bool = True
) -> System:
    def potential(q: Position) -> Scalar:
        return -log_probability_fn(q)

    def kinetic(state: KineticState, _: Position, p: Momentum) -> Scalar:
        assert isinstance(state, EuclideanKineticState)
        p, _ = ravel_pytree(p)
        tril = state.tril
        if tril.ndim == 1:
            alpha = tril * p
        else:
            alpha = jnp.dot(tril.T, p)
        return 0.5 * jnp.sum(jnp.square(alpha))

    def sample_momentum(
        state: KineticState, q: Position, rng_key: random.KeyArray
    ) -> Momentum:
        assert isinstance(state, EuclideanKineticState)
        q, unravel = ravel_pytree(q)
        tril = state.tril
        eps = random.normal(rng_key, q.shape)
        if tril.ndim == 1:
            return unravel(eps / tril)
        return unravel(
            jsp.linalg.solve_triangular(tril, eps, trans=1, lower=True)
        )

    def kinetic_tune_init(size: int) -> KineticState:
        shape = (size,) if diagonal else (size, size)
        mu = jnp.zeros(shape[-1])
        m2 = jnp.zeros(shape)
        return EuclideanKineticState(count=0, tril=jnp.eye(size), mu=mu, m2=m2)

    def kinetic_tune_update(state: KineticState, q: Position) -> KineticState:
        assert isinstance(state, EuclideanKineticState)
        n = state.count + 1
        d1 = q - state.mu
        mu = state.mu + d1 / n
        d2 = q - mu
        if state.m2.ndim == 1:
            m2 = state.m2 + d1 * d2
        else:
            m2 = state.m2 + jnp.outer(d2, d1)
        return EuclideanKineticState(count=n, tril=state.tril, mu=mu, m2=m2)

    def kinetic_tune_finish(state: KineticState) -> KineticState:
        assert isinstance(state, EuclideanKineticState)
        cov = state.m2 / (state.count - 1)
        if state.m2.ndim == 2:
            tril = jsp.linalg.cholesky(cov, lower=True)
        else:
            tril = jnp.sqrt(cov)
        return EuclideanKineticState(
            count=0,
            tril=tril,
            mu=jnp.zeros_like(state.mu),
            m2=jnp.zeros_like(state.m2),
        )

    integrator_init, integrator_update = leapfrog(potential, kinetic)

    return System(
        potential=potential,
        kinetic=kinetic,
        sample_momentum=sample_momentum,
        kinetic_tune_init=kinetic_tune_init,
        kinetic_tune_update=kinetic_tune_update,
        kinetic_tune_finish=kinetic_tune_finish,
        integrator_init=integrator_init,
        integrator_update=integrator_update,
    )


def riemannian(
    log_probability_fn: Callable[[Position], Scalar],
    metric_fn: Callable[[Position], Array],
) -> System:
    def potential(q: Position) -> Scalar:
        return -log_probability_fn(q)

    def kinetic(_: KineticState, q: Position, p: Momentum) -> Scalar:
        p, _ = ravel_pytree(p)
        metric = metric_fn(q)
        tril, _ = jsp.linalg.cho_factor(metric, lower=True)
        half_log_det = jnp.sum(jnp.log(jnp.diag(tril)))
        alpha = jsp.linalg.solve_triangular(tril, p, lower=True)
        return 0.5 * jnp.sum(jnp.square(alpha)) + p.size * half_log_det

    def sample_momentum(
        _: KineticState, q: Position, rng_key: random.KeyArray
    ) -> Momentum:
        metric = metric_fn(q)
        q, unravel = ravel_pytree(q)
        metric = metric_fn(q)
        tril, _ = jsp.linalg.cho_factor(metric, lower=True)
        eps = random.normal(rng_key, q.shape)
        return unravel(jnp.dot(tril, eps))

    # This metric doesn't have any tuning parameters
    def kinetic_tune_init(_: int) -> KineticState:
        return KineticState()

    def kinetic_tune_update(state: KineticState, q: Position) -> KineticState:
        return KineticState()

    def kinetic_tune_finish(state: KineticState) -> KineticState:
        return KineticState()

    integrator_init, integrator_update = implicit_midpoint(potential, kinetic)

    return System(
        potential=potential,
        kinetic=kinetic,
        sample_momentum=sample_momentum,
        kinetic_tune_init=kinetic_tune_init,
        kinetic_tune_update=kinetic_tune_update,
        kinetic_tune_finish=kinetic_tune_finish,
        integrator_init=integrator_init,
        integrator_update=integrator_update,
    )
