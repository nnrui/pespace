"""Utilities shared by the sampler interfaces in ``pespace.inference``.

- :func:`map_fn`: map a function over the sample/particle axis, optionally in
  sequential batches.
- :class:`Sine` / :class:`Cosine`: numpyro prior distributions with sine /
  cosine densities, used for inclination and ecliptic latitude.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints


def map_fn(fn, batch_size: int = 0):
    """Map ``fn`` over the sample/particle axis, batched or chunked.

    ``batch_size=0`` keeps the plain ``jax.vmap`` behaviour; a positive
    ``batch_size`` switches to ``jax.lax.map``, which processes the samples in
    sequential batches of that size.  This bounds the peak memory of the
    likelihood intermediates at the cost of cross-batch parallelism.
    """
    if batch_size > 0:
        return lambda xs: jax.lax.map(fn, xs, batch_size=batch_size)
    return jax.vmap(fn)


class Sine(dist.Distribution):
    """Prior density ``p(x) = sin(x) / 2`` on ``[0, pi]`` (inclination)."""

    support = constraints.interval(0.0, jnp.pi)
    arg_constraints = {}

    def __init__(self, validate_args=None):
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = jax.random.uniform(key, sample_shape)
        return jnp.arccos(1.0 - 2.0 * u)  # inverse CDF of sin(x)/2

    def log_prob(self, value):
        return jnp.where(
            (value >= 0.0) & (value <= jnp.pi),
            jnp.log(0.5 * jnp.sin(value)),
            -jnp.inf,
        )


class Cosine(dist.Distribution):
    """Prior density ``p(x) = cos(x) / 2`` on ``[-pi/2, pi/2]`` (latitude)."""

    support = constraints.interval(-jnp.pi / 2.0, jnp.pi / 2.0)
    arg_constraints = {}

    def __init__(self, validate_args=None):
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = jax.random.uniform(key, sample_shape)
        return jnp.arcsin(2.0 * u - 1.0)  # inverse CDF of cos(x)/2

    def log_prob(self, value):
        return jnp.where(
            (value >= -jnp.pi / 2.0) & (value <= jnp.pi / 2.0),
            jnp.log(0.5 * jnp.cos(value)),
            -jnp.inf,
        )
