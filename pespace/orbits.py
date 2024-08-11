from abc import ABC, abstractmethod
from functools import cached_property

import taichi as ti
import taichi.math as tm
import numpy as np

from .constants import *
from .utils import vec3


# orbit vectors of detector constellation in SSB
OrbitVectorStruct = ti.types.struct(
    n1=vec3,  # unit vectors of link23, in SSB coordinate
    n2=vec3,  # unit vectors of link31, in SSB coordinate
    n3=vec3,  # unit vectors of link12, in SSB coordinate
    p1_det=vec3,  # vector of the node 1 relative to the center of costellation, p1 = p0 + p1_det, in SSB coordinate, in unit of sec
    p2_det=vec3,  # vector of the node 2 relative to the center of costellation, p2 = p0 + p2_det, in SSB coordinate, in unit of sec
    p3_det=vec3,  # vector of the node 3 relative to the center of costellation, p3 = p0 + p3_det, in SSB coordinate, in unit of sec
    p0=vec3,  # vector of the center of the costellation, in SSB coordinate, in unit of sec
)


class OrbitModel(ABC):
    @abstractmethod
    def orbit_vectors(self, time: ti.f64) -> OrbitVectorStruct:
        pass

    # Since currently only analystic Keplerian orbit models where the armlength is a
    # constant are implemented, we define the armlength as a attribute. This may change
    # in future if the armlength is dependent on time.
    @property
    @abstractmethod
    def arm_length_sec(self) -> float:
        pass


class KeplerianGeocentric(OrbitModel):
    """Used for Tianqin"""

    def __init__(
        self,
        arm_length: float,
        rotation_initial: float = 0.0,
        revolution_initial: float = 0.0,
    ) -> None:
        self.arm_length = arm_length
        self.rotation_initial = rotation_initial
        self.revolution_initial = revolution_initial

    pass


@ti.data_oriented
class KaplerianHeliocentric(OrbitModel):
    """Used for LISA, Taiji"""

    # Useful constant
    AU_sec = AU_SI / C_SI
    sqrt3 = tm.sqrt(3)

    def __init__(
        self,
        arm_length: float,
        rotation_initial: float = 0.0,
        revolution_initial: float = 0.0,
    ) -> None:
        """
        https://lisa-ldc.lal.in2p3.fr/static/data/pdf/LDC-manual-Sangria.pdf

        arm_length:
            Arm length of the detector, in the unit of metter.
        rotation_initial:
            The initial phase of detector rotation around its center, [0, 2*pi], default: 0
        revolution_initial:
            The initial phase of detector revolution around the sun, [0, 2*pi], default: 0
        """
        self.arm_length = arm_length
        self.rotation_initial = rotation_initial
        self.revolution_initial = revolution_initial

        # e = L/(2*AU*sqrt3)
        self._ae_sec = self.arm_length_sec / (2 * self.sqrt3)
        # omega: revolution angular velocity of the constellation
        self._omega = 2 * PI / YEAR_SI
        # beta_n: phase for each node
        _beta_1 = self.rotation_initial
        _beta_2 = 2 * PI / 3 + self.rotation_initial
        _beta_3 = 4 * PI / 3 + self.rotation_initial
        self._cb1 = tm.cos(_beta_1)
        self._sb1 = tm.sin(_beta_1)
        self._cb2 = tm.cos(_beta_2)
        self._sb2 = tm.sin(_beta_2)
        self._cb3 = tm.cos(_beta_3)
        self._sb3 = tm.sin(_beta_3)

    # avoiding the performance cost induced by normal property()
    @cached_property
    def arm_length_sec(self) -> float:
        return self.arm_length / C_SI

    @ti.func
    def orbit_vectors(self, time: ti.f64) -> OrbitVectorStruct:
        # alpha: revolution ortial phase
        alpha = self._omega * time + self.revolution_initial
        ca = tm.cos(alpha)
        sa = tm.sin(alpha)
        # vectors of each node in the detector-center-ecliptic coordinate
        # pn_det = pn_ssb - p0_ssb
        node1_det = self._ae_sec * vec3(
            [
                sa * ca * self._sb1 - (1 + sa * sa) * self._cb1,
                sa * ca * self._cb1 - (1 + ca * ca) * self._sb1,
                -self.sqrt3 * (ca * self._cb1 + sa * self._sb1),
            ]
        )
        node2_det = self._ae_sec * vec3(
            [
                sa * ca * self._sb2 - (1 + sa * sa) * self._cb2,
                sa * ca * self._cb2 - (1 + ca * ca) * self._sb2,
                -self.sqrt3 * (ca * self._cb2 + sa * self._sb2),
            ]
        )
        node3_det = self._ae_sec * vec3(
            [
                sa * ca * self._sb3 - (1 + sa * sa) * self._cb3,
                sa * ca * self._cb3 - (1 + ca * ca) * self._sb3,
                -self.sqrt3 * (ca * self._cb3 + sa * self._sb3),
            ]
        )

        return OrbitVectorStruct(
            n1=(node3_det - node2_det) / self.arm_length_sec,
            n2=(node1_det - node3_det) / self.arm_length_sec,
            n3=(node2_det - node1_det) / self.arm_length_sec,
            p1_det=node1_det,
            p2_det=node2_det,
            p3_det=node3_det,
            p0=vec3([ca, sa, 0.0]) * self.AU_sec,
        )


available_orbit_models = {
    "LISA_analytic": KaplerianHeliocentric(2.5e9, 0.0, -PI / 9),
    "Taiji_analytic": KaplerianHeliocentric(3.0e9, 0.0, PI / 9),
    "Tianqin_analytic": KeplerianGeocentric(1.0e8, 0.0, 0.0),
}
