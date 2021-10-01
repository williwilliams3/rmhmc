import dataclasses
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rmhmc.base_types import Array, Momentum, Position, Scalar
from rmhmc.hamiltonian import (
    System,
    compute_total_energy,
    euclidean,
    integrate,
    integrate_trajectory,
    riemannian,
)
from rmhmc.integrator import IntegratorState


def sho(use_euclidean: bool) -> System:
    def log_posterior(q: Position) -> Scalar:
        return -0.5 * jnp.sum(q ** 2)

    def metric(q: Position) -> Array:
        return jnp.diag(jnp.ones_like(q))

    if use_euclidean:
        return euclidean(log_posterior)
    return riemannian(log_posterior, metric)


def planet(use_euclidean: bool) -> System:
    def log_posterior(q: Position) -> Scalar:
        return 1.0 / jnp.sqrt(jnp.sum(q ** 2))

    def metric(q: Position) -> Array:
        return jnp.diag(jnp.ones_like(q))

    if use_euclidean:
        return euclidean(log_posterior)
    return riemannian(log_posterior, metric)


def banana_problem(fixed: bool, use_euclidean: bool) -> System:
    t = 0.5
    sigma_y = 2.0
    sigma_theta = 2.0
    num_obs = 100

    random = np.random.default_rng(1234)
    theta = np.array([t, np.sqrt(1.0 - t)])
    y = (
        theta[0]
        + np.square(theta[1])
        + sigma_y * random.normal(size=(num_obs,))
    )

    def log_posterior(q: Position) -> Scalar:
        p = q[0] + jnp.square(q[1])
        ll = jnp.sum(jnp.square(y - p)) / sigma_y ** 2
        lp = jnp.sum(jnp.square(theta)) / sigma_theta ** 2
        return -0.5 * (ll + lp)

    if fixed:

        def metric(q: Position) -> Array:
            return 10 * jnp.diag(jnp.ones_like(q))

    else:

        def metric(q: Position) -> Array:
            n = y.size
            s = 2.0 * n * q[1] / sigma_y ** 2
            return jnp.array(
                [
                    [n / sigma_y ** 2 + 1.0 / sigma_theta ** 2, s],
                    [
                        s,
                        4.0 * n * jnp.square(q[1]) / sigma_y ** 2
                        + 1.0 / sigma_theta ** 2,
                    ],
                ]
            )

    if fixed and use_euclidean:
        return euclidean(log_posterior)
    return riemannian(log_posterior, metric)


@dataclasses.dataclass(frozen=True)
class Problem:
    builder: Callable[[], System]
    q: Position
    p: Momentum
    num_steps: int
    step_size: float
    energy_prec: float = 1e-4
    pos_prec: float = 5e-5


PROBLEMS = dict(
    sho_riemannian=Problem(
        partial(sho, False),
        jnp.array([0.1]),
        jnp.array([2.0]),
        2000,
        0.01,
    ),
    sho_euclidean=Problem(
        partial(sho, True),
        jnp.array([0.1]),
        jnp.array([2.0]),
        2000,
        0.01,
    ),
    planet_riemannian=Problem(
        partial(planet, False),
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        2000,
        0.01,
        pos_prec=5e-4,
    ),
    planet_euclidean=Problem(
        partial(planet, True),
        jnp.array([1.0, 0.0]),
        jnp.array([0.0, 1.0]),
        2000,
        0.01,
    ),
    banana_riemannian=Problem(
        partial(banana_problem, False, False),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
    ),
    banana_fixed=Problem(
        partial(banana_problem, True, False),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
    ),
    banana_euclidean=Problem(
        partial(banana_problem, True, True),
        jnp.array([0.1, 0.3]),
        jnp.array([2.0, 0.5]),
        2000,
        0.001,
        energy_prec=0.002,
    ),
)


def run(
    system: System, num_steps: int, step_size: float, q: Position, p: Momentum
) -> Tuple[Scalar, Array, IntegratorState, Array]:
    kinetic_state = system.kinetic_tune_init(q.size)
    state = system.integrator_init(kinetic_state, q, p)

    calc_energy = partial(compute_total_energy, system, kinetic_state)
    initial_energy = calc_energy(q, p)

    trace, success = integrate_trajectory(
        system, num_steps, step_size, kinetic_state, state
    )

    energy = jax.vmap(calc_energy)(trace.q, trace.p)
    return initial_energy, energy, trace, success


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_energy_conservation(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore
    initial_energy, energy, _, _ = jax.jit(
        partial(run, system, problem.num_steps, problem.step_size)
    )(problem.q, problem.p)
    np.testing.assert_allclose(
        energy, initial_energy, atol=problem.energy_prec
    )


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_reversibility(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore
    func = jax.jit(partial(run, system, problem.num_steps, problem.step_size))
    _, _, trace, _ = func(problem.q, problem.p)
    _, _, rev_trace, _ = func(trace.q[-1], -trace.p[-1])
    np.testing.assert_allclose(
        trace.q[:-1][::-1], rev_trace.q[:-1], atol=problem.pos_prec
    )


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_volume_conservation(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore
    kinetic_state = system.kinetic_tune_init(problem.q.size)
    phi = jax.jit(
        partial(
            integrate,
            system,
            problem.num_steps,
            problem.step_size,
            kinetic_state,
        )
    )

    qs = []
    ps = []

    eps = 1e-6
    N = problem.q.size
    for n in range(N):
        delta = 0.5 * eps * jnp.eye(N, 1, -n)[:, 0]
        print(delta)

        q = problem.q + delta
        plus, _ = phi(system.integrator_init(kinetic_state, q, problem.p))
        q = problem.q - delta
        minus, _ = phi(system.integrator_init(kinetic_state, q, problem.p))
        qs.append((plus.q - minus.q) / eps)
        ps.append((plus.p - minus.p) / eps)

        p = problem.p + delta
        plus, _ = phi(system.integrator_init(kinetic_state, problem.q, p))
        p = problem.p - delta
        minus, _ = phi(system.integrator_init(kinetic_state, problem.q, p))
        qs.append((plus.q - minus.q) / eps)
        ps.append((plus.p - minus.p) / eps)

    F = jnp.concatenate(
        (jnp.stack(qs, axis=0), jnp.stack(ps, axis=0)), axis=-1
    )
    _, ld = jnp.linalg.slogdet(F)
    np.testing.assert_allclose(ld, 0.0, atol=1e-4)


@pytest.mark.parametrize("problem_name", sorted(PROBLEMS.keys()))
def test_integrate(problem_name: str) -> None:
    problem = PROBLEMS[problem_name]
    system = problem.builder()  # type: ignore

    kinetic_state = system.kinetic_tune_init(problem.q.size)
    state = system.integrator_init(kinetic_state, problem.q, problem.p)
    final_state, success = jax.jit(
        partial(
            integrate,
            system,
            problem.num_steps,
            problem.step_size,
            kinetic_state,
        )
    )(state)
    assert success

    trajectory, success = jax.jit(
        partial(
            integrate_trajectory,
            system,
            problem.num_steps,
            problem.step_size,
            kinetic_state,
        )
    )(state)
    assert np.all(success)

    for v, t in zip(final_state, trajectory):
        np.testing.assert_allclose(v, t[-1])
