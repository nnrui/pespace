from abc import ABC, abstractmethod
from functools import cached_property

import taichi as ti
import taichi.math as tm

from .constants import *
from .utils import vec3


# orbit vectors of detector constellation in ecliptic coordinate
OrbitVectorStruct = ti.types.struct(
    n1=vec3,  # unit vector of link2->3
    n2=vec3,  # unit vector of link3->1
    n3=vec3,  # unit vector of link1->2
    p1_det=vec3,  # vector of the node 1 relative to the center of costellation, in unit of sec, p1 = p0 + p1_det
    p2_det=vec3,  # vector of the node 2 relative to the center of costellation, in unit of sec, p2 = p0 + p2_det
    p3_det=vec3,  # vector of the node 3 relative to the center of costellation, in unit of sec, p3 = p0 + p3_det
    p0=vec3,  # vector of the center of the costellation relative to the sun, in unit of sec
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
    """The analystic Keplerian geocentric orbit model, used for Tianqin."""

    # Useful constant
    AU_sec = AU_SI / C_SI
    sqrt3 = tm.sqrt(3)

    def __init__(
        self,
        arm_length: float,
        rotation_initial: float = 0.0,
        revolution_initial: float = 0.0,
        omega_rotation: float = 2 * PI / (3.65 * DAY_SI),
        lambda_ref: float = 120.5 / 180 * PI,
        beta_ref: float = -4.7 / 180 * PI,
    ) -> None:
        """
        https://doi.org/10.1088/1361-6382/aab52f

        arm_length:
            Arm length of the detector, in the unit of metter.
        rotation_initial:
            The initial phase of detector rotation around its center, [0, 2*pi], default: 0.
        revolution_initial:
            The initial phase of detector revolution around the sun, [0, 2*pi], default: 0.
        omega_rotation:
            Angular velocity of detector rotation, rad s^-1. For Tianqin, 2*PI/(3.65*DAY_SI).
        lambda_ref:
            Ecliptic longitude of the reference source in rad. For RX J0806.3+1527, 120.5/180*PI.
        beta_ref:
            Ecliptic latitude of the reference source in rad. For RX J0806.3+1527, -4.7/180*PI.
        """

        self.arm_length = arm_length
        # radius of detector rotation orbit
        self.r_det_sec = self.arm_length_sec / self.sqrt3

        self.rotation_initial = rotation_initial
        self.revolution_initial = revolution_initial

        self.omega_rotation = omega_rotation
        self.omega_revolution = 2 * PI / YEAR_SI

        self.slam_ref = tm.sin(lambda_ref)
        self.clam_ref = tm.cos(lambda_ref)
        self.sbeta_ref = tm.sin(beta_ref)
        self.cbeta_ref = tm.cos(beta_ref)

    # avoiding the performance cost induced by normal property()
    @cached_property
    def arm_length_sec(self) -> float:
        return self.arm_length / C_SI

    @ti.func
    def orbit_vectors(self, time: ti.f64) -> OrbitVectorStruct:
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
        # pn_det = pn - p0
        node1_det = self.r_det_sec * vec3(
            [
                self.sbeta_ref * self.clam_ref * sk1 + self.slam_ref * ck1,
                self.sbeta_ref * self.slam_ref * sk1 - self.clam_ref * ck1,
                -self.cbeta_ref * sk1,
            ]
        )
        node2_det = self.r_det_sec * vec3(
            [
                self.sbeta_ref * self.clam_ref * sk2 + self.slam_ref * ck2,
                self.sbeta_ref * self.slam_ref * sk2 - self.clam_ref * ck2,
                -self.cbeta_ref * sk2,
            ]
        )
        node3_det = self.r_det_sec * vec3(
            [
                self.sbeta_ref * self.clam_ref * sk3 + self.slam_ref * ck3,
                self.sbeta_ref * self.slam_ref * sk3 - self.clam_ref * ck3,
                -self.cbeta_ref * sk3,
            ]
        )

        return OrbitVectorStruct(
            n1=(node3_det - node2_det) / self.arm_length_sec,
            n2=(node1_det - node3_det) / self.arm_length_sec,
            n3=(node2_det - node1_det) / self.arm_length_sec,
            p1_det=node1_det,
            p2_det=node2_det,
            p3_det=node3_det,
            p0=vec3([tm.cos(alpha), tm.sin(alpha), 0.0]) * self.AU_sec,
        )


@ti.data_oriented
class KaplerianHeliocentric(OrbitModel):
    """The analystic Keplerian heliocentric orbit model, used for LISA, Taiji."""

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

        # r'=AU*ecc=L/(2*sqrt3), ecc=L/(2*AU*sqrt3)
        self.r_prime = self.arm_length_sec / (2 * self.sqrt3)
        # omega: revolution angular velocity of the constellation
        self.omega = 2 * PI / YEAR_SI
        # kappa_n: phase for each node
        kappa_1 = self.rotation_initial
        kappa_2 = kappa_1 + 2 * PI / 3
        kappa_3 = kappa_2 + 2 * PI / 3
        self.ck1 = tm.cos(kappa_1)
        self.sk1 = tm.sin(kappa_1)
        self.ck2 = tm.cos(kappa_2)
        self.sk2 = tm.sin(kappa_2)
        self.ck3 = tm.cos(kappa_3)
        self.sk3 = tm.sin(kappa_3)

    # avoiding the performance cost induced by normal property()
    @cached_property
    def arm_length_sec(self) -> float:
        return self.arm_length / C_SI

    @ti.func
    def orbit_vectors(self, time: ti.f64) -> OrbitVectorStruct:
        # alpha: revolution ortial phase
        alpha = self.omega * time + self.revolution_initial
        ca = tm.cos(alpha)
        sa = tm.sin(alpha)
        # vectors of each node in the detector-center-ecliptic coordinate
        # pn_det = pn_ssb - p0_ssb
        node1_det = self.r_prime * vec3(
            [
                sa * ca * self.sk1 - (1 + sa * sa) * self.ck1,
                sa * ca * self.ck1 - (1 + ca * ca) * self.sk1,
                -self.sqrt3 * (ca * self.ck1 + sa * self.sk1),
            ]
        )
        node2_det = self.r_prime * vec3(
            [
                sa * ca * self.sk2 - (1 + sa * sa) * self.ck2,
                sa * ca * self.ck2 - (1 + ca * ca) * self.sk2,
                -self.sqrt3 * (ca * self.ck2 + sa * self.sk2),
            ]
        )
        node3_det = self.r_prime * vec3(
            [
                sa * ca * self.sk3 - (1 + sa * sa) * self.ck3,
                sa * ca * self.ck3 - (1 + ca * ca) * self.sk3,
                -self.sqrt3 * (ca * self.ck3 + sa * self.sk3),
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
