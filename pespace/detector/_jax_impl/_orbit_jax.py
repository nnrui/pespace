"""Jax implementation of the detector.orbit module."""

from collections import namedtuple

import jax
import jax.numpy as jnp

from ..orbit import OrbitModelBase
from ...utils.constants import *

AU_sec = AU_SI / C_SI
sqrt3 = jnp.sqrt(3)

# orbit vectors of detector constellation in ecliptic coordinate
# similar the ConstellationVectorStruct in detector.orbit
ConstellationVectorTuple = namedtuple(
    "ConstellationVectorTuple", ["n1", "n2", "n3", "x1", "x2", "x3"]
)


@jax.tree_util.register_pytree_node_class
class _KeplerianGeocentricJax(OrbitModelBase):
    """JAX backend for Keplerian geocentric orbit vector computation."""

    def __init__(self, orbit_params):
        self._params = orbit_params

    def tree_flatten(self):
        children = ()
        aux = (self._params,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (params,) = aux
        return cls(params)

    def get_constellation_vectors(self, time):
        """Compute constellation vectors at given times (JAX).

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        ConstellationVectorTuple
            Containing:
            - n1, n2, n3: Unit vectors along the three links;
            - x1, x2, x3: Position vectors of the three nodes in SSB frame (in seconds);
        """
        params = self._params
        alpha = params.omega_revolution * time + params.revolution_initial
        kappa_1 = params.omega_rotation * time + params.rotation_initial
        kappa_2 = kappa_1 + 2.0 * PI / 3.0
        kappa_3 = kappa_2 + 2.0 * PI / 3.0
        ck1, sk1 = jnp.cos(kappa_1), jnp.sin(kappa_1)
        ck2, sk2 = jnp.cos(kappa_2), jnp.sin(kappa_2)
        ck3, sk3 = jnp.cos(kappa_3), jnp.sin(kappa_3)

        node1_det = params.r_det_sec * jnp.stack(
            [
                params.sbeta_ref * params.clam_ref * sk1 + params.slam_ref * ck1,
                params.sbeta_ref * params.slam_ref * sk1 - params.clam_ref * ck1,
                -params.cbeta_ref * sk1,
            ],
            axis=-1,
        )
        node2_det = params.r_det_sec * jnp.stack(
            [
                params.sbeta_ref * params.clam_ref * sk2 + params.slam_ref * ck2,
                params.sbeta_ref * params.slam_ref * sk2 - params.clam_ref * ck2,
                -params.cbeta_ref * sk2,
            ],
            axis=-1,
        )
        node3_det = params.r_det_sec * jnp.stack(
            [
                params.sbeta_ref * params.clam_ref * sk3 + params.slam_ref * ck3,
                params.sbeta_ref * params.slam_ref * sk3 - params.clam_ref * ck3,
                -params.cbeta_ref * sk3,
            ],
            axis=-1,
        )
        x0 = (
            jnp.stack([jnp.cos(alpha), jnp.sin(alpha), jnp.zeros_like(alpha)], axis=-1)
            * AU_sec
        )

        return ConstellationVectorTuple(
            n1=(node3_det - node2_det) / params.static_arm_sec,
            n2=(node1_det - node3_det) / params.static_arm_sec,
            n3=(node2_det - node1_det) / params.static_arm_sec,
            x1=node1_det + x0,
            x2=node2_det + x0,
            x3=node3_det + x0,
        )

    def arm_length_sec(self, time):
        """Return the constant arm-length in seconds.

        For analytic Keplerian orbits, the arm-length is a constant. The ``time``
        parameter is retained only for interface consistency with :class:`OrbitModelBase`.
        The call overhead could be eliminated by the backend of taichi or jax.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        float
            Arm-length in seconds.
        """
        return self._params.static_arm_sec


@jax.tree_util.register_pytree_node_class
class _KaplerianHeliocentricJax(OrbitModelBase):
    """JAX backend for Keplerian heliocentric orbit vector computation."""

    def __init__(self, params):
        self._params = params

    def tree_flatten(self):
        children = ()
        aux = (self._params,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (params,) = aux
        return cls(params)

    def get_constellation_vectors(self, time):
        """Compute constellation vectors at given times (JAX).

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        ConstellationVectorTuple
            Containing:
            - n1, n2, n3: Unit vectors along the three links;
            - x1, x2, x3: Position vectors of the three nodes in SSB frame (in seconds);
        """
        params = self._params
        alpha = params.omega * time + params.revolution_initial
        ca, sa = jnp.cos(alpha), jnp.sin(alpha)
        ca_pow2, sa_pow2 = ca * ca, sa * sa

        x1_det = params.r_prime * jnp.stack(
            [
                sa * ca * params.sk1 - (1.0 + sa_pow2) * params.ck1,
                sa * ca * params.ck1 - (1.0 + ca_pow2) * params.sk1,
                -sqrt3 * (ca * params.ck1 + sa * params.sk1),
            ],
            axis=-1,
        )
        x2_det = params.r_prime * jnp.stack(
            [
                sa * ca * params.sk2 - (1.0 + sa_pow2) * params.ck2,
                sa * ca * params.ck2 - (1.0 + ca_pow2) * params.sk2,
                -sqrt3 * (ca * params.ck2 + sa * params.sk2),
            ],
            axis=-1,
        )
        x3_det = params.r_prime * jnp.stack(
            [
                sa * ca * params.sk3 - (1.0 + sa_pow2) * params.ck3,
                sa * ca * params.ck3 - (1.0 + ca_pow2) * params.sk3,
                -sqrt3 * (ca * params.ck3 + sa * params.sk3),
            ],
            axis=-1,
        )
        x0 = jnp.stack([ca, sa, jnp.zeros_like(ca)], axis=-1) * AU_sec

        return ConstellationVectorTuple(
            n1=(x3_det - x2_det) / params.static_arm_sec,
            n2=(x1_det - x3_det) / params.static_arm_sec,
            n3=(x2_det - x1_det) / params.static_arm_sec,
            x1=x1_det + x0,
            x2=x2_det + x0,
            x3=x3_det + x0,
        )

    def arm_length_sec(self, time):
        """Return the constant arm-length in seconds.

        For analytic Keplerian orbits, the arm-length is a constant. The ``time``
        parameter is retained only for interface consistency with :class:`OrbitModelBase`.
        The call overhead could be eliminated by the backend of taichi or jax.

        Parameters
        ----------
        time : float
            Time in seconds.

        Returns
        -------
        float
            Arm-length in seconds.
        """
        return self._params.static_arm_sec
