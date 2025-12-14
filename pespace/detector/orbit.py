from abc import ABC, abstractmethod
from functools import cached_property

import taichi as ti
import taichi.math as tm

from ..utils.constants import *


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
    @abstractmethod
    def update_detector_vectors(self, out: ti.template(), time: float):
        pass

    # Since currently only analystic Keplerian orbit models where the armlength is a
    # constant are implemented, we define the armlength as a attribute. This may change
    # in future if the armlength is dependent on time.
    @property
    @abstractmethod
    def arm_length_sec(self) -> float:
        pass


@ti.data_oriented
class KeplerianGeocentric(OrbitModelBase):
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

    @cached_property
    def arm_length_sec(self) -> float:
        return self.arm_length / C_SI

    @ti.func
    def update_detector_vectors(self, out: ti.template(), time: float):
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

        # set the elements of vectors manually to avoid the assertion failure of !operand->is<AllocaStmt>()
        # vectors of each node in the detector-center-ecliptic coordinate
        # xn_det = xn - x0
        node1_det = ti.Vector([0.0, 0.0, 0.0])
        node1_det[0] = self.r_det_sec * (
            self.sbeta_ref * self.clam_ref * sk1 + self.slam_ref * ck1
        )
        node1_det[1] = self.r_det_sec * (
            self.sbeta_ref * self.slam_ref * sk1 - self.clam_ref * ck1
        )
        node1_det[2] = self.r_det_sec * (-self.cbeta_ref * sk1)

        node2_det = ti.Vector([0.0, 0.0, 0.0])
        node2_det[0] = self.r_det_sec * (
            self.sbeta_ref * self.clam_ref * sk2 + self.slam_ref * ck2
        )
        node2_det[1] = self.r_det_sec * (
            self.sbeta_ref * self.slam_ref * sk2 - self.clam_ref * ck2
        )
        node2_det[2] = self.r_det_sec * (-self.cbeta_ref * sk2)

        node3_det = ti.Vector([0.0, 0.0, 0.0])
        node3_det[0] = self.r_det_sec * (
            self.sbeta_ref * self.clam_ref * sk3 + self.slam_ref * ck3
        )
        node3_det[1] = self.r_det_sec * (
            self.sbeta_ref * self.slam_ref * sk3 - self.clam_ref * ck3
        )
        node3_det[2] = self.r_det_sec * (-self.cbeta_ref * sk3)

        x0 = ti.Vector([0.0, 0.0, 0.0])
        x0[0] = self.AU_sec * tm.cos(alpha)
        x0[1] = self.AU_sec * tm.sin(alpha)
        x0[2] = 0.0

        for i in ti.static(range(3)):
            out.n1[i] = (node3_det[i] - node2_det[i]) / self.arm_length_sec
            out.n2[i] = (node1_det[i] - node3_det[i]) / self.arm_length_sec
            out.n3[i] = (node2_det[i] - node1_det[i]) / self.arm_length_sec
            out.x1[i] = node1_det[i] + x0[i]
            out.x2[i] = node2_det[i] + x0[i]
            out.x3[i] = node3_det[i] + x0[i]


@ti.data_oriented
class KaplerianHeliocentric(OrbitModelBase):
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
        arm_length:
            Arm length of the detector, in the unit of meter.
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

    @cached_property
    def arm_length_sec(self) -> float:
        return self.arm_length / C_SI

    @ti.func
    def update_detector_vectors(self, out: ti.template(), time: float):
        # alpha: revolution ortial phase
        alpha = self.omega * time + self.revolution_initial
        ca = tm.cos(alpha)
        sa = tm.sin(alpha)
        ca_pow2 = ca * ca
        sa_pow2 = sa * sa

        # set the elements of vectors manually to avoid the assertion failure of !operand->is<AllocaStmt>()
        # vectors of each spacecraft in the solar system barycentric coordinate
        # xn_ssb = xn_det_ssb + x0_ssb
        x1_det = ti.Vector([0.0, 0.0, 0.0])
        x1_det[0] = self.r_prime * (sa * ca * self.sk1 - (1 + sa_pow2) * self.ck1)
        x1_det[1] = self.r_prime * (sa * ca * self.ck1 - (1 + ca_pow2) * self.sk1)
        x1_det[2] = self.r_prime * (-self.sqrt3 * (ca * self.ck1 + sa * self.sk1))

        x2_det = ti.Vector([0.0, 0.0, 0.0])
        x2_det[0] = self.r_prime * (sa * ca * self.sk2 - (1 + sa_pow2) * self.ck2)
        x2_det[1] = self.r_prime * (sa * ca * self.ck2 - (1 + ca_pow2) * self.sk2)
        x2_det[2] = self.r_prime * (-self.sqrt3 * (ca * self.ck2 + sa * self.sk2))

        x3_det = ti.Vector([0.0, 0.0, 0.0])
        x3_det[0] = self.r_prime * (sa * ca * self.sk3 - (1 + sa_pow2) * self.ck3)
        x3_det[1] = self.r_prime * (sa * ca * self.ck3 - (1 + ca_pow2) * self.sk3)
        x3_det[2] = self.r_prime * (-self.sqrt3 * (ca * self.ck3 + sa * self.sk3))

        x0 = ti.Vector([0.0, 0.0, 0.0])
        x0[0] = self.AU_sec * ca
        x0[1] = self.AU_sec * sa
        x0[2] = 0.0

        for i in ti.static(range(3)):
            out.n1[i] = (x3_det[i] - x2_det[i]) / self.arm_length_sec
            out.n2[i] = (x1_det[i] - x3_det[i]) / self.arm_length_sec
            out.n3[i] = (x2_det[i] - x1_det[i]) / self.arm_length_sec
            out.x1[i] = x1_det[i] + x0[i]
            out.x2[i] = x2_det[i] + x0[i]
            out.x3[i] = x3_det[i] + x0[i]

    # @ti.func
    # def update_detector_vectors(self, out: ti.template(), time: float):
    #     # alpha: revolution ortial phase
    #     alpha = self.omega * time + self.revolution_initial
    #     ca = tm.cos(alpha)
    #     sa = tm.sin(alpha)
    #     ca_pow2 = ca * ca
    #     sa_pow2 = sa * sa

    #     # set the elements of vectors manually to avoid the assertion failure of !operand->is<AllocaStmt>()
    #     # vectors of each spacecraft in the solar system barycentric coordinate
    #     # xn_ssb = xn_det_ssb + x0_ssb
    #     x1_det = ti.Vector([0.0, 0.0, 0.0])
    #     x1_det[0] = self.r_prime * (sa * ca * self.sk1 - (1 + sa_pow2) * self.ck1)
    #     x1_det[1] = self.r_prime * (sa * ca * self.ck1 - (1 + ca_pow2) * self.sk1)
    #     x1_det[2] = self.r_prime * (-self.sqrt3 * (ca * self.ck1 + sa * self.sk1))

    #     x2_det = ti.Vector([0.0, 0.0, 0.0])
    #     x2_det[0] = self.r_prime * (sa * ca * self.sk2 - (1 + sa_pow2) * self.ck2)
    #     x2_det[1] = self.r_prime * (sa * ca * self.ck2 - (1 + ca_pow2) * self.sk2)
    #     x2_det[2] = self.r_prime * (-self.sqrt3 * (ca * self.ck2 + sa * self.sk2))

    #     x3_det = ti.Vector([0.0, 0.0, 0.0])
    #     x3_det[0] = self.r_prime * (sa * ca * self.sk3 - (1 + sa_pow2) * self.ck3)
    #     x3_det[1] = self.r_prime * (sa * ca * self.ck3 - (1 + ca_pow2) * self.sk3)
    #     x3_det[2] = self.r_prime * (-self.sqrt3 * (ca * self.ck3 + sa * self.sk3))

    #     x0 = ti.Vector([0.0, 0.0, 0.0])
    #     x0[0] = self.AU_sec * ca
    #     x0[1] = self.AU_sec * sa
    #     x0[2] = 0.0

    #     for i in ti.static(range(3)):
    #         out[None].n1[i] = (x3_det[i] - x2_det[i]) / self.arm_length_sec
    #         out[None].n2[i] = (x1_det[i] - x3_det[i]) / self.arm_length_sec
    #         out[None].n3[i] = (x2_det[i] - x1_det[i]) / self.arm_length_sec
    #         out[None].x1[i] = x1_det[i] + x0[i]
    #         out[None].x2[i] = x2_det[i] + x0[i]
    #         out[None].x3[i] = x3_det[i] + x0[i]


available_orbit_models = {
    "LISA_analytic": KaplerianHeliocentric(2.5e9, 0.0, -PI / 9),
    "Taiji_analytic": KaplerianHeliocentric(3.0e9, 0.0, PI / 9),
    "Tianqin_analytic": KeplerianGeocentric(1.0e8, 0.0, 0.0),
}
