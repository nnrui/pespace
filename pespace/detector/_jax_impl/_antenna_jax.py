from collections import namedtuple

import jax.numpy as jnp


PolarizationTuple = namedtuple("PolarizationTuple", ["plus", "cross"])

def get_polarization_tensor_ssb_jax(lam, beta, psi):
    """Compute GW polarization tensors in SSB frame (JAX).

    Parameters
    ----------
    lam : float
        Ecliptic longitude in radians.
    beta : float
        Ecliptic latitude in radians.
    psi : float
        Polarization angle in radians.

    Returns
    -------
    dict[str, jnp.ndarray]
        'plus' and 'cross' keys, each a 3x3 matrix.
    """
    sin_lam, cos_lam = jnp.sin(lam), jnp.cos(lam)
    sin_beta, cos_beta = jnp.sin(beta), jnp.cos(beta)
    sin_psi, cos_psi = jnp.sin(psi), jnp.cos(psi)
    p = jnp.array(
        [
            sin_lam * cos_psi - cos_lam * sin_beta * sin_psi,
            -cos_lam * cos_psi - sin_lam * sin_beta * sin_psi,
            cos_beta * sin_psi,
        ]
    )
    q = jnp.array(
        [
            -sin_lam * sin_psi - cos_lam * sin_beta * cos_psi,
            cos_lam * sin_psi - sin_lam * sin_beta * cos_psi,
            cos_beta * cos_psi,
        ]
    )
    return PolarizationTuple(
        plus=jnp.outer(p, p) - jnp.outer(q, q),
        cross=jnp.outer(p, q) + jnp.outer(q, p),
    )


def get_gw_propagation_unit_vector_jax(lam, beta):
    """Compute GW propagation unit vector in SSB frame (JAX).

    Parameters
    ----------
    lam : float
        Ecliptic longitude in radians.
    beta : float
        Ecliptic latitude in radians.

    Returns
    -------
    jnp.ndarray
        Shape (3,) unit vector pointing from source to SSB origin.
    """
    return jnp.array(
        [-jnp.cos(beta) * jnp.cos(lam), -jnp.cos(beta) * jnp.sin(lam), -jnp.sin(beta)]
    )
