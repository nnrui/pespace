# from __future__ import annotations
# since the type hint in current taichi-lang does not support to parse types from strings,
# use string literal types for foward reference in python scope.
import weakref
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
import taichi as ti
import taichi.math as tm

from .orbit import OrbitModelBase, available_orbit_models, ConstellationVectorStruct
from ..utils.utils import (
    taichi_field_to_complex_numpy_array_dict,
    complex_numpy_array_dict_to_taichi_field,
    get_polarization_tensor_ssb,
    get_gw_propagation_unit_vector,
    get_pattern_function,
    sinc,
    noise_weighted_inner_product,
    ti_complex,
    PolarizationStruct,
    SingleLinkStructComplex,
    SingleLinkStructReal,
    INTERPOLATE_KERNELS,
)
from ..utils.constants import *


@ti.data_oriented
class InterferometerAntenna:
    # TODO:
    # - include suppot to higher modes
    # - add more tdi combination, and setting it as input argument (or including it in TDIChannelData class)

    def __init__(
        self,
        name: str,
        tdi_data: "TDIChannelData",
        orbit_model: str | OrbitModelBase,
        response_model: "SingleLinkResponseModel",
        tdi_combination: "TDICombinationModel",
        needs_grad: bool = False,
        needs_dual: bool = False,
    ) -> None:
        """ """
        self.name = name
        self.tdi_data = tdi_data
        if isinstance(orbit_model, OrbitModelBase):
            self.orbit_model = orbit_model
        elif isinstance(orbit_model, str):
            try:
                self.orbit_model = available_orbit_models[orbit_model]
            except KeyError:
                raise ValueError(
                    f"{orbit_model} is not a implemented orbit model. \n"
                    f"Current available models are {[*available_orbit_models.keys()]}"
                )
        else:
            raise TypeError(
                f"Expected OrbitModelBase or str for orbit_model, but got {type(orbit_model).__name__}."
            )
        self.response_model = response_model
        self.tdi_combination = tdi_combination

        self.needs_grad = needs_grad
        self.needs_dual = needs_dual

        self.params = ti.Struct.field(
            dict(lam=float, beta=float, psi=float, tc=float),
            shape=(),
            needs_grad=self.needs_grad,
            needs_dual=self.needs_dual,
        )
        self.tdi_response = None
        self.single_link_response = None

        # note the length of single_link_response in TD depending on tdi combination model, init the response_model after the tdi_combination
        self.tdi_combination.init_tdi_combination_model(self)
        self.response_model.init_single_link_response_model(self)

    def update_input_params(
        self,
        lam: float,
        beta: float,
        psi: float,
        tc: float,
    ) -> None:
        self.params[None].lam = lam
        self.params[None].beta = beta
        self.params[None].psi = psi
        self.params[None].tc = tc

    def update_detector_response(
        self,
        waveform: ti.StructField,
        lam: float,
        beta: float,
        psi: float,
        tc: float,
    ) -> None:
        # for auto-diff, reading params field must be done in ti scope
        self.update_input_params(lam, beta, psi, tc)
        self.response_model.update_single_link_response(waveform)
        self.tdi_combination.update_tdi_response()

    def inject_signal(
        self,
        waveform: ti.StructField,
        lam: float,
        beta: float,
        psi: float,
        tc: float,
    ) -> None:
        self.update_detector_response(waveform, lam, beta, psi, tc)
        if self.tdi_combination.domain == "fd":
            self.tdi_data.add_into_fd_data(self.tdi_response)
        elif self.tdi_combination.domain == "td":
            self.tdi_data.add_into_td_data(self.tdi_response)
        elif self.tdi_combination.domain == "wd":
            self.tdi_data.add_into_wd_data(self.tdi_response)
        else:
            raise NotImplementedError(
                f"Unsupported domain {self.tdi_combination.domain}, only 'fd', 'td' or "
                "'wd' are supported currently."
            )

    def get_optimal_snr(
        self,
        waveform: ti.StructField,
        lam: float,
        beta: float,
        psi: float,
        tc: float,
    ) -> dict[str, float]:
        pass

    @property
    def tdi_response_numpy(self) -> dict[str, NDArray]:
        return taichi_field_to_complex_numpy_array_dict(self.tdi_response)

    @property
    def single_link_response_numpy(self) -> dict[str, NDArray]:
        return taichi_field_to_complex_numpy_array_dict(self.single_link_response)


class SingleLinkResponseModel(ABC):

    @abstractmethod
    def update_single_link_response(self) -> None:
        pass


@ti.data_oriented
class FDResponseModelMarset2018(SingleLinkResponseModel):

    def init_single_link_response_model(self, detector: InterferometerAntenna) -> None:
        self.detector = weakref.proxy(detector)
        self.detector.single_link_response = SingleLinkStructComplex.field(
            shape=(self.detector.tdi_data.data_info.frequency_series_length,),
            needs_grad=self.detector.needs_grad,
            needs_dual=self.detector.needs_dual,
        )

        self._geometry_terms = ti.Struct.field(
            dict(pol_tensor=PolarizationStruct, prop_direc=ti.types.vector(3, float)),
            shape=(),
            needs_grad=self.detector.needs_grad,
            needs_dual=self.detector.needs_dual,
        )
        self._det_vectors = ConstellationVectorStruct.field(
            shape=(self.detector.tdi_data.data_info.frequency_series_length,),
            needs_grad=self.detector.needs_grad,
            needs_dual=self.detector.needs_dual,
        )
        self._pattern_funcs = ti.Struct.field(
            dict(
                n1_plus=float,
                n1_cross=float,
                n2_plus=float,
                n2_cross=float,
                n3_plus=float,
                n3_cross=float,
            ),
            shape=(self.detector.tdi_data.data_info.frequency_series_length,),
            needs_grad=self.detector.needs_grad,
            needs_dual=self.detector.needs_dual,
        )

    def update_single_link_response(self, waveform: ti.StructField):
        self._update_geometry_terms()
        self._loop_frequencies(waveform)

    @ti.kernel
    def _update_geometry_terms(self):
        sin_lam = tm.sin(self.detector.params[None].lam)
        cos_lam = tm.cos(self.detector.params[None].lam)
        sin_beta = tm.sin(self.detector.params[None].beta)
        cos_beta = tm.cos(self.detector.params[None].beta)
        sin_psi = tm.sin(self.detector.params[None].psi)
        cos_psi = tm.cos(self.detector.params[None].psi)

        # to avoid the assertion failure of !operand->is<AllocaStmt>(), manually unroll
        # the loop to set each vector, may improve in future
        # set the polarization tensor
        p = ti.Vector([0.0] * 3)
        p[0] = sin_lam * cos_psi - cos_lam * sin_beta * sin_psi
        p[1] = -cos_lam * cos_psi - sin_lam * sin_beta * sin_psi
        p[2] = cos_beta * sin_psi
        q = ti.Vector([0.0] * 3)
        q[0] = -sin_lam * sin_psi - cos_lam * sin_beta * cos_psi
        q[1] = cos_lam * sin_psi - sin_lam * sin_beta * cos_psi
        q[2] = cos_beta * cos_psi
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                self._geometry_terms[None].pol_tensor.plus[i, j] = (
                    p[i] * p[j] - q[i] * q[j]
                )
                self._geometry_terms[None].pol_tensor.cross[i, j] = (
                    p[i] * q[j] + q[i] * p[j]
                )
        # set the unit vector of the GW propagation direction
        self._geometry_terms[None].prop_direc[0] = -cos_beta * cos_lam
        self._geometry_terms[None].prop_direc[1] = -cos_beta * sin_lam
        self._geometry_terms[None].prop_direc[2] = -sin_beta

    @ti.kernel
    def _loop_frequencies(self, waveform: ti.template()):
        # put the loop in a seperated kernel for the rule required by auto-diff
        # the waveform_container uses the AOS layout, operating modes in the inner loop
        for i in self.detector.single_link_response:
            fi = self.detector.tdi_data.frequency_samples[i]
            tshift = -2.0 * PI * fi * self.detector.params[None].tc
            cexp_tshift_re = tm.cos(tshift)
            cexp_tshift_im = tm.sin(tshift)

            if ti.static(sorted(waveform.keys) == sorted(["cross", "plus", "tf"])):
                hp_re = (
                    waveform[i].plus[0] * cexp_tshift_re
                    - waveform[i].plus[1] * cexp_tshift_im
                )
                hp_im = (
                    waveform[i].plus[0] * cexp_tshift_im
                    + waveform[i].plus[1] * cexp_tshift_re
                )
                hc_re = (
                    waveform[i].cross[0] * cexp_tshift_re
                    - waveform[i].cross[1] * cexp_tshift_im
                )
                hc_im = (
                    waveform[i].cross[0] * cexp_tshift_im
                    + waveform[i].cross[1] * cexp_tshift_re
                )
                tf = waveform[i].tf + self.detector.params[None].tc

                self.detector.orbit_model.update_detector_vectors(
                    self._det_vectors[i], tf
                )
                self._update_pattern_functions(
                    self._pattern_funcs[i], self._det_vectors[i]
                )

                n1_h_n1_re = 0.0  # ti_complex
                n1_h_n1_im = 0.0  # ti_complex
                n2_h_n2_re = 0.0  # ti_complex
                n2_h_n2_im = 0.0  # ti_complex
                n3_h_n3_re = 0.0  # ti_complex
                n3_h_n3_im = 0.0  # ti_complex
                n1_h_n1_re = (
                    self._pattern_funcs[i].n1_plus * hp_re
                    + self._pattern_funcs[i].n1_cross * hc_re
                )
                n1_h_n1_im = (
                    self._pattern_funcs[i].n1_plus * hp_im
                    + self._pattern_funcs[i].n1_cross * hc_im
                )
                n2_h_n2_re = (
                    self._pattern_funcs[i].n2_plus * hp_re
                    + self._pattern_funcs[i].n2_cross * hc_re
                )
                n2_h_n2_im = (
                    self._pattern_funcs[i].n2_plus * hp_im
                    + self._pattern_funcs[i].n2_cross * hc_im
                )
                n3_h_n3_re = (
                    self._pattern_funcs[i].n3_plus * hp_re
                    + self._pattern_funcs[i].n3_cross * hc_re
                )
                n3_h_n3_im = (
                    self._pattern_funcs[i].n3_plus * hp_im
                    + self._pattern_funcs[i].n3_cross * hc_im
                )

                k_n1 = 0.0  # scalar
                k_n2 = 0.0  # scalar
                k_n3 = 0.0  # scalar
                for n in ti.static(range(3)):
                    k_n1 += (
                        self._geometry_terms[None].prop_direc[n]
                        * self._det_vectors[i].n1[n]
                    )
                    k_n2 += (
                        self._geometry_terms[None].prop_direc[n]
                        * self._det_vectors[i].n2[n]
                    )
                    k_n3 += (
                        self._geometry_terms[None].prop_direc[n]
                        * self._det_vectors[i].n3[n]
                    )

                pi_f_L = PI * fi * self.detector.orbit_model.arm_length_sec
                sinc32 = sinc(pi_f_L * (1.0 - k_n1))
                sinc23 = sinc(pi_f_L * (1.0 + k_n1))
                sinc13 = sinc(pi_f_L * (1.0 - k_n2))
                sinc31 = sinc(pi_f_L * (1.0 + k_n2))
                sinc21 = sinc(pi_f_L * (1.0 - k_n3))
                sinc12 = sinc(pi_f_L * (1.0 + k_n3))

                k_x1_x2 = 0.0  # scalar
                k_x2_x3 = 0.0  # scalar
                k_x3_x1 = 0.0  # scalar
                for n in ti.static(range(3)):
                    k_x1_x2 += self._geometry_terms[None].prop_direc[n] * (
                        self._det_vectors[i].x1[n] + self._det_vectors[i].x2[n]
                    )
                    k_x2_x3 += self._geometry_terms[None].prop_direc[n] * (
                        self._det_vectors[i].x2[n] + self._det_vectors[i].x3[n]
                    )
                    k_x3_x1 += self._geometry_terms[None].prop_direc[n] * (
                        self._det_vectors[i].x3[n] + self._det_vectors[i].x1[n]
                    )

                exp12_re = 0.0
                exp12_im = 0.0
                exp23_re = 0.0
                exp23_im = 0.0
                exp31_re = 0.0
                exp31_im = 0.0
                phi_12 = -PI * fi * (self.detector.orbit_model.arm_length_sec + k_x1_x2)
                phi_23 = -PI * fi * (self.detector.orbit_model.arm_length_sec + k_x2_x3)
                phi_31 = -PI * fi * (self.detector.orbit_model.arm_length_sec + k_x3_x1)
                exp12_re = tm.cos(phi_12)
                exp12_im = tm.sin(phi_12)
                exp23_re = tm.cos(phi_23)
                exp23_im = tm.sin(phi_23)
                exp31_re = tm.cos(phi_31)
                exp31_im = tm.sin(phi_31)

                temp_12_re = -(-pi_f_L) * (
                    n3_h_n3_re * exp12_im + n3_h_n3_im * exp12_re
                )
                temp_12_im = (-pi_f_L) * (n3_h_n3_re * exp12_re - n3_h_n3_im * exp12_im)
                temp_23_re = -(-pi_f_L) * (
                    n1_h_n1_re * exp23_im + n1_h_n1_im * exp23_re
                )
                temp_23_im = (-pi_f_L) * (n1_h_n1_re * exp23_re - n1_h_n1_im * exp23_im)
                temp_31_re = -(-pi_f_L) * (
                    n2_h_n2_re * exp31_im + n2_h_n2_im * exp31_re
                )
                temp_31_im = (-pi_f_L) * (n2_h_n2_re * exp31_re - n2_h_n2_im * exp31_im)

                self.detector.single_link_response[i].link12[0] = sinc12 * temp_12_re
                self.detector.single_link_response[i].link12[1] = sinc12 * temp_12_im

                self.detector.single_link_response[i].link21[0] = sinc21 * temp_12_re
                self.detector.single_link_response[i].link21[1] = sinc21 * temp_12_im

                self.detector.single_link_response[i].link23[0] = sinc23 * temp_23_re
                self.detector.single_link_response[i].link23[1] = sinc23 * temp_23_im

                self.detector.single_link_response[i].link32[0] = sinc32 * temp_23_re
                self.detector.single_link_response[i].link32[1] = sinc32 * temp_23_im

                self.detector.single_link_response[i].link31[0] = sinc31 * temp_31_re
                self.detector.single_link_response[i].link31[1] = sinc31 * temp_31_im

                self.detector.single_link_response[i].link13[0] = sinc13 * temp_31_re
                self.detector.single_link_response[i].link13[1] = sinc13 * temp_31_im
            else:  # waveform with HM does not support FwdMode autodiff currently
                links = SingleLinkStructComplex()
                cexp_tshift = ti.Vector([cexp_tshift_re, cexp_tshift_im])

                for mode in ti.static(waveform.keys):
                    hp_mode = tm.cmul(waveform[i][mode].plus, cexp_tshift)
                    hc_mode = tm.cmul(waveform[i][mode].cross, cexp_tshift)
                    tf_mode = waveform[i][mode].tf + self.detector.params[None].tc
                    links_mode = self._get_responses_at_fi(
                        self._geometry_terms[None].pol_tensor,
                        self._geometry_terms[None].prop_direc,
                        hp_mode,
                        hc_mode,
                        tf_mode,
                        fi,
                    )
                    links.link12 += links_mode.link12
                    links.link21 += links_mode.link21
                    links.link23 += links_mode.link23
                    links.link32 += links_mode.link32
                    links.link31 += links_mode.link31
                    links.link13 += links_mode.link13

                self.detector.single_link_response[i] = links

    @ti.func
    def _get_responses_at_fi(
        self,
        pol_tensor: PolarizationStruct,
        k: ti.types.vector(3, float),
        hp: ti_complex,
        hc: ti_complex,
        tf: float,
        fi: float,
    ) -> SingleLinkStructComplex:
        det_vectors = ConstellationVectorStruct()
        self.detector.orbit_model.update_detector_vectors(det_vectors, tf)
        # n1: unit vector of 2 -> 3
        pattern_p_n1, pattern_c_n1 = get_pattern_function(pol_tensor, det_vectors.n1)
        n1_h_n1 = pattern_p_n1 * hp + pattern_c_n1 * hc  # ti_complex
        # n2: unit vector of 3 -> 1
        pattern_p_n2, pattern_c_n2 = get_pattern_function(pol_tensor, det_vectors.n2)
        n2_h_n2 = pattern_p_n2 * hp + pattern_c_n2 * hc  # ti_complex
        # n3: unit vector of 1 -> 2
        pattern_p_n3, pattern_c_n3 = get_pattern_function(pol_tensor, det_vectors.n3)
        n3_h_n3 = pattern_p_n3 * hp + pattern_c_n3 * hc  # ti_complex

        k_n1 = k.dot(det_vectors.n1)  # scalar
        k_n2 = k.dot(det_vectors.n2)  # scalar
        k_n3 = k.dot(det_vectors.n3)  # scalar

        k_x1_x2 = k.dot(det_vectors.x1 + det_vectors.x2)  # scalar
        k_x2_x3 = k.dot(det_vectors.x2 + det_vectors.x3)  # scalar
        k_x3_x1 = k.dot(det_vectors.x3 + det_vectors.x1)  # scalar

        pi_f_L = PI * fi * self.detector.orbit_model.arm_length_sec  # scalar
        sinc32 = sinc(pi_f_L * (1.0 - k_n1))  # scalar
        sinc23 = sinc(pi_f_L * (1.0 + k_n1))  # scalar
        sinc13 = sinc(pi_f_L * (1.0 - k_n2))  # scalar
        sinc31 = sinc(pi_f_L * (1.0 + k_n2))  # scalar
        sinc21 = sinc(pi_f_L * (1.0 - k_n3))  # scalar
        sinc12 = sinc(pi_f_L * (1.0 + k_n3))  # scalar

        common_exp = -PI * fi * ti_complex([0.0, 1.0])  # ti_complex
        exp12 = tm.cexp(
            common_exp * (self.detector.orbit_model.arm_length_sec + k_x1_x2)
        )  # ti_complex
        exp23 = tm.cexp(
            common_exp * (self.detector.orbit_model.arm_length_sec + k_x2_x3)
        )  # ti_complex
        exp31 = tm.cexp(
            common_exp * (self.detector.orbit_model.arm_length_sec + k_x3_x1)
        )  # ti_complex

        prefactor = -pi_f_L * ti_complex([0.0, 1.0])  # ti_complex
        link12 = sinc12 * tm.cmul(tm.cmul(prefactor, n3_h_n3), exp12)  # ti_complex
        link21 = sinc21 * tm.cmul(tm.cmul(prefactor, n3_h_n3), exp12)  # ti_complex
        link23 = sinc23 * tm.cmul(tm.cmul(prefactor, n1_h_n1), exp23)  # ti_complex
        link32 = sinc32 * tm.cmul(tm.cmul(prefactor, n1_h_n1), exp23)  # ti_complex
        link31 = sinc31 * tm.cmul(tm.cmul(prefactor, n2_h_n2), exp31)  # ti_complex
        link13 = sinc13 * tm.cmul(tm.cmul(prefactor, n2_h_n2), exp31)  # ti_complex

        return SingleLinkStructComplex(link12, link21, link23, link32, link31, link13)

    @ti.func
    def _update_pattern_functions(
        self, pattern_funcs: ti.template(), det_vectors: ti.template()
    ):
        pattern_funcs.n1_plus = 0.0
        pattern_funcs.n1_cross = 0.0
        pattern_funcs.n2_plus = 0.0
        pattern_funcs.n2_cross = 0.0
        pattern_funcs.n3_plus = 0.0
        pattern_funcs.n3_cross = 0.0
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                pattern_funcs.n1_plus += (
                    det_vectors.n1[i]
                    * det_vectors.n1[j]
                    * self._geometry_terms[None].pol_tensor.plus[i, j]
                )
                pattern_funcs.n1_cross += (
                    det_vectors.n1[i]
                    * det_vectors.n1[j]
                    * self._geometry_terms[None].pol_tensor.cross[i, j]
                )
                pattern_funcs.n2_plus += (
                    det_vectors.n2[i]
                    * det_vectors.n2[j]
                    * self._geometry_terms[None].pol_tensor.plus[i, j]
                )
                pattern_funcs.n2_cross += (
                    det_vectors.n2[i]
                    * det_vectors.n2[j]
                    * self._geometry_terms[None].pol_tensor.cross[i, j]
                )
                pattern_funcs.n3_plus += (
                    det_vectors.n3[i]
                    * det_vectors.n3[j]
                    * self._geometry_terms[None].pol_tensor.plus[i, j]
                )
                pattern_funcs.n3_cross += (
                    det_vectors.n3[i]
                    * det_vectors.n3[j]
                    * self._geometry_terms[None].pol_tensor.cross[i, j]
                )

    # @ti.func
    # def _update_pattern_functions(self, pattern_funcs: ti.template(), det_vectors:ti.template()):
    #     self._pattern_funcs[None].n1_plus = 0.0
    #     self._pattern_funcs[None].n1_cross = 0.0
    #     self._pattern_funcs[None].n2_plus = 0.0
    #     self._pattern_funcs[None].n2_cross = 0.0
    #     self._pattern_funcs[None].n3_plus = 0.0
    #     self._pattern_funcs[None].n3_cross = 0.0
    #     for i in ti.static(range(3)):
    #         for j in ti.static(range(3)):
    #             self._pattern_funcs[None].n1_plus += (
    #                 self._det_vectors[None].n1[i]
    #                 * self._det_vectors[None].n1[j]
    #                 * self._geometry_terms[None].pol_tensor.plus[i, j]
    #             )
    #             self._pattern_funcs[None].n1_cross += (
    #                 self._det_vectors[None].n1[i]
    #                 * self._det_vectors[None].n1[j]
    #                 * self._geometry_terms[None].pol_tensor.cross[i, j]
    #             )
    #             self._pattern_funcs[None].n2_plus += (
    #                 self._det_vectors[None].n2[i]
    #                 * self._det_vectors[None].n2[j]
    #                 * self._geometry_terms[None].pol_tensor.plus[i, j]
    #             )
    #             self._pattern_funcs[None].n2_cross += (
    #                 self._det_vectors[None].n2[i]
    #                 * self._det_vectors[None].n2[j]
    #                 * self._geometry_terms[None].pol_tensor.cross[i, j]
    #             )
    #             self._pattern_funcs[None].n3_plus += (
    #                 self._det_vectors[None].n3[i]
    #                 * self._det_vectors[None].n3[j]
    #                 * self._geometry_terms[None].pol_tensor.plus[i, j]
    #             )
    #             self._pattern_funcs[None].n3_cross += (
    #                 self._det_vectors[None].n3[i]
    #                 * self._det_vectors[None].n3[j]
    #                 * self._geometry_terms[None].pol_tensor.cross[i, j]
    #             )

    # @ti.func
    # def _get_responses_at_fi_old(
    #     self,
    #     hp: ti_complex,
    #     hc: ti_complex,
    #     tf: float,
    #     fi: float,
    #     pol_tensor: PolarizationStruct,
    #     k: ti.types.vector(3, float),
    # ) -> SingleLinkStructComplex:
    #     det_vectors = self.detector.orbit_model.get_constellation_vectors(tf)
    #     # n1: unit vector of 2 -> 3
    #     n1_h_n1 = (
    #         det_vectors.n1 @ pol_tensor.plus @ det_vectors.n1 * hp
    #         + det_vectors.n1 @ pol_tensor.cross @ det_vectors.n1 * hc
    #     )  # ti_complex
    #     # n2: unit vector of 3 -> 1
    #     n2_h_n2 = (
    #         det_vectors.n2 @ pol_tensor.plus @ det_vectors.n2 * hp
    #         + det_vectors.n2 @ pol_tensor.cross @ det_vectors.n2 * hc
    #     )  # ti_complex
    #     # n3: unit vector of 1 -> 2
    #     n3_h_n3 = (
    #         det_vectors.n3 @ pol_tensor.plus @ det_vectors.n3 * hp
    #         + det_vectors.n3 @ pol_tensor.cross @ det_vectors.n3 * hc
    #     )  # ti_complex

    #     k_n1 = k @ det_vectors.n1  # scalar
    #     k_n2 = k @ det_vectors.n2  # scalar
    #     k_n3 = k @ det_vectors.n3  # scalar

    #     k_x1_x2 = k @ (det_vectors.x1 + det_vectors.x2)  # scalar
    #     k_x2_x3 = k @ (det_vectors.x2 + det_vectors.x3)  # scalar
    #     k_x3_x1 = k @ (det_vectors.x3 + det_vectors.x1)  # scalar

    #     pi_f_L = PI * fi * self.detector.orbit_model.arm_length_sec  # scalar
    #     sinc32 = sinc(pi_f_L * (1.0 - k_n1))  # scalar
    #     sinc23 = sinc(pi_f_L * (1.0 + k_n1))  # scalar
    #     sinc13 = sinc(pi_f_L * (1.0 - k_n2))  # scalar
    #     sinc31 = sinc(pi_f_L * (1.0 + k_n2))  # scalar
    #     sinc21 = sinc(pi_f_L * (1.0 - k_n3))  # scalar
    #     sinc12 = sinc(pi_f_L * (1.0 + k_n3))  # scalar

    #     common_exp = -PI * fi * ti_complex([0.0, 1.0])  # ti_complex
    #     exp12 = tm.cexp(
    #         common_exp * (self.detector.orbit_model.arm_length_sec + k_x1_x2)
    #     )  # ti_complex
    #     exp23 = tm.cexp(
    #         common_exp * (self.detector.orbit_model.arm_length_sec + k_x2_x3)
    #     )  # ti_complex
    #     exp31 = tm.cexp(
    #         common_exp * (self.detector.orbit_model.arm_length_sec + k_x3_x1)
    #     )  # ti_complex

    #     prefactor = -pi_f_L * ti_complex([0.0, 1.0])  # ti_complex
    #     link12 = sinc12 * tm.cmul(tm.cmul(prefactor, n3_h_n3), exp12)  # ti_complex
    #     link21 = sinc21 * tm.cmul(tm.cmul(prefactor, n3_h_n3), exp12)  # ti_complex
    #     link23 = sinc23 * tm.cmul(tm.cmul(prefactor, n1_h_n1), exp23)  # ti_complex
    #     link32 = sinc32 * tm.cmul(tm.cmul(prefactor, n1_h_n1), exp23)  # ti_complex
    #     link31 = sinc31 * tm.cmul(tm.cmul(prefactor, n2_h_n2), exp31)  # ti_complex
    #     link13 = sinc13 * tm.cmul(tm.cmul(prefactor, n2_h_n2), exp31)  # ti_complex

    #     return SingleLinkStructComplex(link12, link21, link23, link32, link31, link13)

    # @ti.kernel
    # def update_single_link_response_old(
    #     self,
    #     waveform: ti.template(),
    #     lam: float,
    #     beta: float,
    #     psi: float,
    #     tc: float,
    # ):
    #     pol_tensor = get_polarization_tensor_ssb(lam, beta, psi)  # matrix: 3*3
    #     prop_direc = get_gw_propagation_unit_vector(lam, beta)  # vector: 3

    #     # the waveform_container uses the AOS layout, operating modes in the inner loop
    #     for i in self.detector.single_link_response:
    #         fi = self.detector.tdi_data.frequency_samples[i]
    #         cexp_tshift = tm.cexp(ti_complex([0.0, -2.0 * PI * fi * tc]))
    #         links = SingleLinkStructComplex()

    #         if ti.static(sorted(waveform.keys) == sorted(["cross", "plus", "tf"])):
    #             hp = tm.cmul(waveform[i].plus, cexp_tshift)
    #             hc = tm.cmul(waveform[i].cross, cexp_tshift)
    #             tf = waveform[i].tf + tc
    #             links = self._get_responses_at_fi(
    #                 hp, hc, tf, fi, pol_tensor, prop_direc
    #             )
    #         else:
    #             for mode in ti.static(waveform.keys):
    #                 hp_mode = tm.cmul(waveform[i][mode].plus, cexp_tshift)
    #                 hc_mode = tm.cmul(waveform[i][mode].cross, cexp_tshift)
    #                 tf_mode = waveform[i][mode].tf + tc
    #                 links_mode = self._get_responses_at_fi(
    #                     hp_mode, hc_mode, tf_mode, fi, pol_tensor, prop_direc
    #                 )
    #                 links.link12 += links_mode.link12
    #                 links.link21 += links_mode.link21
    #                 links.link23 += links_mode.link23
    #                 links.link32 += links_mode.link32
    #                 links.link31 += links_mode.link31
    #                 links.link13 += links_mode.link13

    #         self.detector.single_link_response[i] = links


class FDResponseModelLongWavelength(SingleLinkResponseModel):
    pass


class FDResponseModelStaticLongWavelength(SingleLinkResponseModel):
    pass


# @ti.data_oriented
# class TDResponseModelCornish2003(SingleLinkResponseModel):

#     def __init__(self, interpolate_kernel: str | tuple[str, int]):
#         """
#         interpolate_kernel:
#         """
#         if isinstance(interpolate_kernel, str) and (interpolate_kernel == "linear"):
#             self.interpolate_kernel = linear_interpolate_kernel
#             self.interpolate_kernel_length = 3
#         elif isinstance(interpolate_kernel, tuple):
#             self.interpolate_kernel = None
#             self.interpolate_kernel_length = interpolate_kernel[1]
#         else:
#             raise

#     def init_single_link_response_model(self, detector: InterferometerAntenna) -> None:
#         self.detector = weakref.proxy(detector)
#         self.detector.single_link_response = SingleLinkStructReal.field(
#             shape=(self.detector.tdi_combination.extended_time_series_length,),
#         )

#         self.extended_time_samples = ti.field(
#             float, shape=(self.detector.tdi_combination.extended_time_series_length,)
#         )
#         added_time_samples = (
#             np.arange(self.detector.tdi_combination.added_time_samples_number)[::-1]
#             * self.detector.tdi_data.data_info.delta_time
#             + self.detector.tdi_data.data_info.start_time
#         )
#         self.extended_time_samples.from_numpy(
#             np.concatenate(
#                 added_time_samples,
#                 self.detector.tdi_data.data_info.time_samples_array,
#             )
#         )


#     @ti.func
#     def _get_shifted_waveform(
#         self, waveform: ti.types.ndarray(dtype=float, ndim=2), time: float
#     ):
#         dt = self.detector.tdi_data.data_info.delta_time
#         idx = time // dt
#         frac = time % dt
#         hp_left, hc_left = waveform[idx, 0], waveform[idx, 1]
#         hp_right, hc_right = waveform[idx + 1, 0], waveform[idx + 1, 1]
#         hp = linear_interpolate(hp_left, hp_right, frac)
#         hc = linear_interpolate(hc_left, hc_right, frac)
#         return hp, hc

#     def _ensure_waveform_length(self, waveform_container:dict[str, NDArray | float],
#                                 tc:float):
#         dt = self.detector.tdi_data.data_info.delta_time
#         ###
#         x_max = self._get_x_max()
#         t_min = self.extended_time_samples[0] - self.detector.orbit_model.armlength_sec - x_max
#         t_max = self.extended_time_samples[-1] + x_max
#         ###
#         wf_t0 = waveform_container['t0']
#         wf_tend = waveform_container["t0"] + waveform_container["data"].shape[0]*dt
#         prepend_length = 0
#         append_length = 0
#         if int((t_min - wf_t0) // dt) < int(self.interpolate_kernel_length//2):
#             padding_


#     def update_single_link_response(
#         self,
#         waveform_container: dict[str, NDArray | float],
#         lam: float,
#         beta: float,
#         psi: float,
#         tc: float,
#     ):
#         waveform, t0 = self._ensure_waveform_length(waveform_container, tc)
#         self.update_single_link_response_kernel(
#             waveform,
#             t0,
#             lam,
#             beta,
#             psi,
#         )

#     @ti.kernel
#     def update_single_link_response_kernel(
#         self,
#         waveform: ti.types.ndarray(dtype=float, ndim=2),
#         t0: float,  # time of the first data point in waveform
#         lam: float,
#         beta: float,
#         psi: float,
#     ):
#         pol_tensor = get_polarization_tensor_ssb(lam, beta, psi)  # matrix: 3*3
#         k = get_gw_propagation_unit_vector(lam, beta)  # vector: 3

#         for i in self.detector.single_link_response:
#             t = self.extended_time_samples[i]

#             constellation_vectors = self.detector.orbit_model.get_constellation_vectors(t)  # fmt: skip

#             k_x1 = k @ constellation_vectors.x1
#             k_x2 = k @ constellation_vectors.x2
#             k_x3 = k @ constellation_vectors.x3

#             L_arm = self.detector.orbit_model.armlength_sec
#             # TODO: handle the case when out of the boundaies of waveform
#             hp_send_x1, hc_send_x1 = self._get_shifted_waveform(
#                 waveform, (t - L_arm - k_x1 - t0)
#             )
#             hp_send_x2, hc_send_x2 = self._get_shifted_waveform(
#                 waveform, (t - L_arm - k_x2 - t0)
#             )
#             hp_send_x3, hc_send_x3 = self._get_shifted_waveform(
#                 waveform, (t - L_arm - k_x3 - t0)
#             )
#             hp_rece_x1, hc_rece_x1 = self._get_shifted_waveform(
#                 waveform, (t - k_x1 - t0)
#             )
#             hp_rece_x2, hc_rece_x2 = self._get_shifted_waveform(
#                 waveform, (t - k_x2 - t0)
#             )
#             hp_rece_x3, hc_rece_x3 = self._get_shifted_waveform(
#                 waveform, (t - k_x3 - t0)
#             )

#             n1_plus_tensor_n1 = (
#                 constellation_vectors.n1 @ pol_tensor.plus @ constellation_vectors.n1
#             )
#             n1_cross_tensor_n1 = (
#                 constellation_vectors.n1 @ pol_tensor.cross @ constellation_vectors.n1
#             )
#             n2_plus_tensor_n2 = (
#                 constellation_vectors.n2 @ pol_tensor.plus @ constellation_vectors.n2
#             )
#             n2_cross_tensor_n2 = (
#                 constellation_vectors.n2 @ pol_tensor.cross @ constellation_vectors.n2
#             )
#             n3_plus_tensor_n3 = (
#                 constellation_vectors.n3 @ pol_tensor.plus @ constellation_vectors.n3
#             )
#             n3_cross_tensor_n3 = (
#                 constellation_vectors.n3 @ pol_tensor.cross @ constellation_vectors.n3
#             )

#             # # n1: unit vector of 2 -> 3
#             # n1_plus_tensor_n1 = (
#             #     constellation_vectors.n1
#             #     @ pol_tensor.plus
#             #     @ constellation_vectors.n1
#             #     * ()
#             #     + constellation_vectors.n1
#             #     @ pol_tensor.cross
#             #     @ constellation_vectors.n1
#             #     * ()
#             # )
#             # # n2: unit vector of 3 -> 1
#             # n2_h_n2 = (
#             #     constellation_vectors.n2
#             #     @ pol_tensor.plus
#             #     @ constellation_vectors.n2
#             #     * hp
#             #     + constellation_vectors.n2
#             #     @ pol_tensor.cross
#             #     @ constellation_vectors.n2
#             #     * hc
#             # )
#             # # n3: unit vector of 1 -> 2
#             # n3_h_n3 = (
#             #     constellation_vectors.n3
#             #     @ pol_tensor.plus
#             #     @ constellation_vectors.n3
#             #     * hp
#             #     + constellation_vectors.n3
#             #     @ pol_tensor.cross
#             #     @ constellation_vectors.n3
#             #     * hc
#             # )

#             k_n1 = k @ constellation_vectors.n1
#             k_n2 = k @ constellation_vectors.n2
#             k_n3 = k @ constellation_vectors.n3

#             self.detector.single_link_response[i].link12 = (
#                 0.5
#                 * (
#                     n3_plus_tensor_n3 * (h_send_x2.plus - h_rece_x1.plus)
#                     + n3_cross_tensor_n3 * (hc_send_x2 - hc_rece_x1)
#                 )
#                 / (1.0 + k_n3)
#             )
#             self.detector.single_link_response[i].link21 = (
#                 0.5
#                 * (
#                     n3_plus_tensor_n3 * (hp_send_x1 - hp_rece_x2)
#                     + n3_cross_tensor_n3 * (hc_send_x1 - hc_rece_x2)
#                 )
#                 / (1.0 - k_n3)
#             )
#             self.detector.single_link_response[i].link23 = (
#                 0.5
#                 * (
#                     n1_plus_tensor_n1 * (hp_send_x3 - hp_rece_x2)
#                     + n1_cross_tensor_n1 * (hc_send_x3 - hc_rece_x2)
#                 )
#                 / (1.0 + k_n1)
#             )
#             self.detector.single_link_response[i].link32 = (
#                 0.5
#                 * (
#                     n1_plus_tensor_n1 * (hp_send_x2 - hp_rece_x3)
#                     + n1_cross_tensor_n1 * (hc_send_x2 - hc_rece_x3)
#                 )
#                 / (1.0 - k_n1)
#             )
#             self.detector.single_link_response[i].link31 = (
#                 0.5
#                 * (
#                     n2_plus_tensor_n2 * (hp_send_x1 - hp_rece_x3)
#                     + n2_cross_tensor_n2 * (hc_send_x1 - hc_rece_x3)
#                 )
#                 / (1.0 + k_n2)
#             )
#             self.detector.single_link_response[i].link13 = (
#                 0.5
#                 * (
#                     n2_plus_tensor_n2 * (hp_send_x3 - hp_rece_x1)
#                     + n2_cross_tensor_n2 * (hc_send_x3 - hc_rece_x1)
#                 )
#                 / (1.0 - k_n2)
#             )


########################################################################################
# def optimal_snr(self):
#     '''
#     compute the optimal SNR of the GW signal of each channels

#     Returns:
#     ========
#     dict, contain snr of each channels, if ("A", "E", "T") or ("A", "E") channels are contained, total SNR also will be returned
#     '''
#     if self.signals is None:
#         raise Exception('the signals in None, set the GW signal before computing SNR')

#     indep_chan = sorted([chan for chan in self.TDI_channels if chan in ['A', 'E', 'T']])
#     compute_total = (indep_chan == ['A', 'E', 'T'] or indep_chan == ['A', 'E'])
#     if compute_total:
#         total_rho2 = 0.0
#     else:
#         print(f'TDI channels are set to {self.TDI_channels} which don\'t contain independent channels '
#                '("A", "E", "T") or ("A", "E") total SNR will not be computed.')

#     snr_dict = {}
#     for chan in self.TDI_channels:
#         rho2_chan = noise_weighted_inner_product(self.signals[chan], self.signals[chan], self.psd_array[chan], self.delta_f)
#         snr_dict[chan] = rho2_chan**0.5
#         if chan in indep_chan and compute_total:
#             total_rho2 += rho2_chan

#     if compute_total:
#         snr_dict['total'] = total_rho2**0.5

#     return snr_dict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tdi import TDIChannelData, TDICombinationModel
