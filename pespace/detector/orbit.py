"""Module for orbit models, used to define a specific detector."""

from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import taichi as ti
import taichi.math as tm

from ..utils.constants import *

AU_sec = AU_SI / C_SI
sqrt3 = np.sqrt(3)

# orbit vectors of detector constellation in ecliptic coordinate
# fellowing the convention of 2108.01167
ConstellationVectorStruct = ti.types.struct(
    n1=ti.types.vector(3, float),  # unit vector of link2->3 (n32) in SSB
    n2=ti.types.vector(3, float),  # unit vector of link3->1 (n13) in SSB
    n3=ti.types.vector(3, float),  # unit vector of link1->2 (n21) in SSB
    x1=ti.types.vector(3, float),  # vector of the node 1 in SSB, in unit of sec, x1 = x0 + x1_det # fmt: skip
    x2=ti.types.vector(3, float),  # vector of the node 2 in SSB, in unit of sec, x2 = x0 + x2_det # fmt: skip
    x3=ti.types.vector(3, float),  # vector of the node 3 in SSB, in unit of sec, x3 = x0 + x3_det # fmt: skip
)


class OrbitModelBase(ABC):
    """Abstract base class for orbit models of detector constellations."""

    @abstractmethod
    def get_constellation_vectors(self, time: float):
        """
        Compute constellation vectors used in response computation at a given time.

        Parameters
        ----------
        time : float
            Time in seconds at which to compute the constellation vectors.

        Returns
        -------
        ConstellationVectorStruct (taichi) | ConstellationVectorTuple (jax)
            Structure containing unit vectors and position vectors of the three nodes.
        """
        pass

    @abstractmethod
    def arm_length_sec(self, time: float):
        """
        Get the arm length in units of seconds.

        Returns
        -------
        float
            Arm length in seconds (light travel time).

        Notes
        -----
        Currently only analytic Keplerian orbit models where the armlength is a constant
        are implemented. Orbital models with higher precision may be implemented in future,
        where arm lengths could be time-dependent. Therefore, we define arm-length as a
        time-dependent function in the base class to facilitate future extensions.
        """
        pass


class KeplerianGeocentric:
    """
    Analytical Keplerian geocentric orbit model, used for Tianqin.

    References
    ----------
    .. [1] https://doi.org/10.1088/1361-6382/aab52f
    """

    def __init__(
        self,
        arm_length: float,
        rotation_initial: float = 0.0,
        revolution_initial: float = 0.0,
        omega_rotation: float = 2 * PI / (3.65 * DAY_SI),
        lambda_ref: float = 120.5 / 180 * PI,
        beta_ref: float = -4.7 / 180 * PI,
        backend: str = "taichi",
    ) -> None:
        """
        Initialize the Keplerian geocentric orbit model.

        Parameters
        ----------
        arm_length : float
            Arm length of the detector in meters.
        rotation_initial : float, optional
            Initial phase of detector rotation around its center in radians,
            range :math:`[0, 2\\pi]`. Default is 0.0.
        revolution_initial : float, optional
            Initial phase of detector revolution around the Sun in radians,
            range :math:`[0, 2\\pi]`. Default is 0.0.
        omega_rotation : float, optional
            Angular velocity of detector rotation in rad/s.
            Default is :math:`2\\pi/(3.65\\text{days})` for TianQin.
        lambda_ref : float, optional
            Ecliptic longitude of the reference source in radians.
            Default is :math:`120.5/180*\\pi` for RX J0806.3+1527.
        beta_ref : float, optional
            Ecliptic latitude of the reference source in radians.
            Default is :math:`-4.7/180*\\pi` for RX J0806.3+1527.
        backend : str, optional
            Computation backend, either 'taichi' (default) or 'jax'.
        """
        OrbitParams = namedtuple(
            "OrbitParams",
            [
                "static_arm_sec",
                "r_det_sec",
                "rotation_initial",
                "revolution_initial",
                "omega_rotation",
                "omega_revolution",
                "slam_ref",
                "clam_ref",
                "sbeta_ref",
                "cbeta_ref",
            ],
        )

        orbit_params = OrbitParams(
            static_arm_sec=arm_length / C_SI,
            r_det_sec=arm_length / C_SI / np.sqrt(3),  # radius of detector rotation orbit # fmt: skip
            rotation_initial=rotation_initial,
            revolution_initial=revolution_initial,
            omega_rotation=omega_rotation,
            omega_revolution=2 * PI / YEAR_SI,
            slam_ref=np.sin(lambda_ref),
            clam_ref=np.cos(lambda_ref),
            sbeta_ref=np.sin(beta_ref),
            cbeta_ref=np.cos(beta_ref),
        )
        self.orbit_params = orbit_params

        if backend == "jax":
            from ._jax_impl._orbit_jax import _KeplerianGeocentricJax

            self._backend_impl = _KeplerianGeocentricJax(orbit_params)
        elif backend == "taichi":
            self._backend_impl = _KeplerianGeocentricTaichi(orbit_params)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.get_constellation_vectors = self._backend_impl.get_constellation_vectors
        self.arm_length_sec = self._backend_impl.arm_length_sec


class KaplerianHeliocentric:
    """
    Analytical Keplerian heliocentric orbit model, used for LISA and Taiji.

    References
    ----------
    .. [1] https://arxiv.org/abs/2301.02967
    .. [2] https://lisa-ldc.in2p3.fr/static/data/pdf/LDC-manual-Sangria.pdf
    """

    def __init__(
        self,
        arm_length: float,
        rotation_initial: float = 0.0,
        revolution_initial: float = 0.0,
        backend: str = "taichi",
    ) -> None:
        """
        Initialize the Keplerian heliocentric orbit model.

        Parameters
        ----------
        arm_length : float
            Arm length of the detector in meters.
        rotation_initial : float, optional
            Initial phase of detector rotation around its center in radians,
            range :math:`[0, 2\\pi]`. Default is 0.0.
        revolution_initial : float, optional
            Initial phase of detector revolution around the Sun in radians,
            range :math:`[0, 2\\pi]`. Default is 0.0.
        backend : str, optional
            Computation backend, either 'taichi' (default) or 'jax'.
        """
        # static arm-length
        arm_sec = arm_length / C_SI
        # r'=AU*ecc=L/(2*sqrt3), ecc=L/(2*AU*sqrt3)
        r_prime = arm_sec / (2 * np.sqrt(3))
        # omega: revolution angular velocity of the constellation
        omega = 2 * PI / YEAR_SI
        # kappa_n: phase for each node
        kappa_1 = rotation_initial
        kappa_2 = kappa_1 + 2 * PI / 3
        kappa_3 = kappa_2 + 2 * PI / 3

        OrbitParams = namedtuple(
            "OrbitParams",
            [
                "static_arm_sec",
                "rotation_initial",
                "revolution_initial",
                "r_prime",
                "omega",
                "ck1",
                "sk1",
                "ck2",
                "sk2",
                "ck3",
                "sk3",
            ],
        )
        orbit_params = OrbitParams(
            static_arm_sec=arm_sec,
            rotation_initial=rotation_initial,
            revolution_initial=revolution_initial,
            r_prime=r_prime,
            omega=omega,
            ck1=np.cos(kappa_1),
            sk1=np.sin(kappa_1),
            ck2=np.cos(kappa_2),
            sk2=np.sin(kappa_2),
            ck3=np.cos(kappa_3),
            sk3=np.sin(kappa_3),
        )
        self.orbit_params = orbit_params

        if backend == "jax":
            from ._jax_impl._orbit_jax import _KaplerianHeliocentricJax

            self._backend_impl = _KaplerianHeliocentricJax(orbit_params)
        elif backend == "taichi":
            self._backend_impl = _KaplerianHeliocentricTaichi(orbit_params)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.get_constellation_vectors = self._backend_impl.get_constellation_vectors
        self.arm_length_sec = self._backend_impl.arm_length_sec


@ti.data_oriented
class _KeplerianGeocentricTaichi(OrbitModelBase):
    """Taichi backend for Keplerian geocentric orbit vector computation."""

    def __init__(self, orbit_params):
        self.static_arm_sec = orbit_params.static_arm_sec
        self.r_det_sec = orbit_params.r_det_sec
        self.rotation_initial = orbit_params.rotation_initial
        self.revolution_initial = orbit_params.revolution_initial
        self.omega_rotation = orbit_params.omega_rotation
        self.omega_revolution = orbit_params.omega_revolution
        self.slam_ref = orbit_params.slam_ref
        self.clam_ref = orbit_params.clam_ref
        self.sbeta_ref = orbit_params.sbeta_ref
        self.cbeta_ref = orbit_params.cbeta_ref

    @ti.func
    def get_constellation_vectors(self, time: float) -> ConstellationVectorStruct:
        """Compute constellation vectors at a given time (taichi).

        Only callable from within ``@ti.kernel`` functions.

        Parameters
        ----------
        time : float
            Time in seconds at which to compute the constellation vectors.

        Returns
        -------
        ConstellationVectorStruct
            Containing:
            - n1, n2, n3: Unit vectors along the three links;
            - x1, x2, x3: Position vectors of the three nodes in SSB frame (in seconds);
        """
        # alpha: revolution ortial phase
        alpha = self.omega_revolution * time + self.revolution_initial
        # kappa_n: rotaion phase for each node
        kappa_1 = self.omega_rotation * time + self.rotation_initial
        kappa_2 = kappa_1 + 2 * PI / 3
        kappa_3 = kappa_2 + 2 * PI / 3
        ck1 = tm.cos(kappa_1)
        sk1 = tm.sin(kappa_1)
        ck2 = tm.cos(kappa_2)
        sk2 = tm.sin(kappa_2)
        ck3 = tm.cos(kappa_3)
        sk3 = tm.sin(kappa_3)
        # vectors of each node in the detector-center-ecliptic coordinate
        # xn_det = xn - x0
        node1_det = self.r_det_sec * ti.Vector(
            [
                self.sbeta_ref * self.clam_ref * sk1 + self.slam_ref * ck1,
                self.sbeta_ref * self.slam_ref * sk1 - self.clam_ref * ck1,
                -self.cbeta_ref * sk1,
            ]
        )
        node2_det = self.r_det_sec * ti.Vector(
            [
                self.sbeta_ref * self.clam_ref * sk2 + self.slam_ref * ck2,
                self.sbeta_ref * self.slam_ref * sk2 - self.clam_ref * ck2,
                -self.cbeta_ref * sk2,
            ]
        )
        node3_det = self.r_det_sec * ti.Vector(
            [
                self.sbeta_ref * self.clam_ref * sk3 + self.slam_ref * ck3,
                self.sbeta_ref * self.slam_ref * sk3 - self.clam_ref * ck3,
                -self.cbeta_ref * sk3,
            ]
        )
        x0 = ti.Vector([tm.cos(alpha), tm.sin(alpha), 0.0]) * AU_sec

        return ConstellationVectorStruct(
            n1=(node3_det - node2_det) / self.static_arm_sec,
            n2=(node1_det - node3_det) / self.static_arm_sec,
            n3=(node2_det - node1_det) / self.static_arm_sec,
            x1=node1_det + x0,
            x2=node2_det + x0,
            x3=node3_det + x0,
        )

    @ti.func
    def arm_length_sec(self, time) -> float:
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
        return self.static_arm_sec


@ti.data_oriented
class _KaplerianHeliocentricTaichi(OrbitModelBase):
    """Taichi backend for Keplerian heliocentric orbit vector computation."""

    def __init__(self, orbit_params):
        self.static_arm_sec = orbit_params.static_arm_sec
        self.rotation_initial = orbit_params.rotation_initial
        self.revolution_initial = orbit_params.revolution_initial
        self.r_prime = orbit_params.r_prime
        self.omega = orbit_params.omega
        self.ck1 = orbit_params.ck1
        self.sk1 = orbit_params.sk1
        self.ck2 = orbit_params.ck2
        self.sk2 = orbit_params.sk2
        self.ck3 = orbit_params.ck3
        self.sk3 = orbit_params.sk3

    @ti.func
    def get_constellation_vectors(self, time: float) -> ConstellationVectorStruct:
        """Compute constellation vectors at a given time (taichi).

        Only callable from within ``@ti.kernel`` functions.

        Parameters
        ----------
        time : float
            Time in seconds at which to compute the constellation vectors.

        Returns
        -------
        ConstellationVectorStruct
            Containing:
            - n1, n2, n3: Unit vectors along the three links;
            - x1, x2, x3: Position vectors of the three nodes in SSB frame (in seconds);

        """
        # alpha: revolution ortial phase
        alpha = self.omega * time + self.revolution_initial
        ca = tm.cos(alpha)
        sa = tm.sin(alpha)
        ca_pow2 = ca * ca
        sa_pow2 = sa * sa
        # vectors of each spacecraft in the solar system barycentric coordinate
        # xn_ssb = xn_det_ssb + x0_ssb
        x1_det = self.r_prime * ti.Vector(
            [
                sa * ca * self.sk1 - (1 + sa_pow2) * self.ck1,
                sa * ca * self.ck1 - (1 + ca_pow2) * self.sk1,
                -sqrt3 * (ca * self.ck1 + sa * self.sk1),
            ]
        )
        x2_det = self.r_prime * ti.Vector(
            [
                sa * ca * self.sk2 - (1 + sa_pow2) * self.ck2,
                sa * ca * self.ck2 - (1 + ca_pow2) * self.sk2,
                -sqrt3 * (ca * self.ck2 + sa * self.sk2),
            ]
        )
        x3_det = self.r_prime * ti.Vector(
            [
                sa * ca * self.sk3 - (1 + sa_pow2) * self.ck3,
                sa * ca * self.ck3 - (1 + ca_pow2) * self.sk3,
                -sqrt3 * (ca * self.ck3 + sa * self.sk3),
            ]
        )
        x0 = ti.Vector([ca, sa, 0.0]) * AU_sec

        return ConstellationVectorStruct(
            n1=(x3_det - x2_det) / self.static_arm_sec,
            n2=(x1_det - x3_det) / self.static_arm_sec,
            n3=(x2_det - x1_det) / self.static_arm_sec,
            x1=x1_det + x0,
            x2=x2_det + x0,
            x3=x3_det + x0,
        )

    @ti.func
    def arm_length_sec(self, time) -> float:
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
        return self.static_arm_sec


available_orbit_models = {
    "LISA_analytic": KaplerianHeliocentric(2.5e9, 0.0, -PI / 9),
    "Taiji_analytic": KaplerianHeliocentric(3.0e9, 0.0, PI / 9),
    "Tianqin_analytic": KeplerianGeocentric(1.0e8, 0.0, 0.0),
}
