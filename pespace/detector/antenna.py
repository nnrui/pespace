# from __future__ import annotations
# since the type hint in current taichi-lang does not support to parse types from strings,
# use string literal types for foward reference in python scope.
import weakref
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
import taichi as ti
import taichi.math as tm

from .orbit import OrbitModelBase, available_orbit_models
from ..utils.utils import (
    taichi_field_to_complex_numpy_array_dict,
    complex_numpy_array_dict_to_taichi_field,
    get_polarization_tensor_ssb,
    get_gw_propagation_unit_vector,
    sinc,
    noise_weighted_inner_product,
    ComplexNumber,
    SingleLinkStructComplex,
    SingleLinkStructReal,
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

        self.tdi_response = None
        self.single_link_response = None

        # note the length of single_link_response in TD depending on tdi combination model, init the response_model after the tdi_combination
        self.tdi_combination.init_tdi_combination_model(self)
        self.response_model.init_single_link_response_model(self)

    def update_detector_response(
        self,
        waveform: ti.StructField,
        lam: float,
        beta: float,
        psi: float,
        tc: float,
    ) -> None:
        self.response_model.update_single_link_response(waveform, lam, beta, psi, tc)
        self.tdi_combination.update_tdi_response()

    def inject_signal(
        self,
        waveform: ti.StructField,
        lam: float,
        beta: float,
        psi: float,
        tc: float,
    ) -> None:
        pass

    @property
    def tdi_response_numpy(self) -> dict[str, NDArray[np.complex128]]:
        return taichi_field_to_complex_numpy_array_dict(self.tdi_response)

    @property
    def single_link_response_numpy(self) -> dict[str, NDArray[np.complex128]]:
        return taichi_field_to_complex_numpy_array_dict(self.single_link_response)

    # def _init_fd_response_container(self) -> None:
    #     if self.TDI_data.data_info is None:
    #         raise ValueError(
    #             "The `data_info` of the passed-in TDI_data is `None`. Can not \
    #             determine the frequency series length. Please set it before calling \
    #             `initilize_response_container_in_frequency_domain`."
    #         )
    #     else:
    #         self.response_container = ti.Struct.field(
    #             dict.fromkeys(self.TDI_data.data_info.channels, ComplexNumber),
    #             shape=(self.TDI_data.data_info.frequency_series_length,),
    #         )
    #         self._FD_response_assistance = ti.Struct.field(
    #             dict(
    #                 delay_factor=ComplexNumber,
    #                 TDI_generation_prefactor=ComplexNumber,
    #                 single_links=SingleLinksStruct,
    #             ),
    #             shape=(self.TDI_data.data_info.frequency_series_length,),
    #         )
    #         self._compute_TDI_prefactor_for_FD_response()

    # @ti.kernel
    # def _compute_TDI_prefactor_for_FD_response(self):
    #     # maybe better moving this func into TDI generation class, since the 1-z^2 or z^2-1 depending on the calculation of tdi combination
    #     for i in self.TDI_data.frequency_samples:
    #         z = tm.cexp(
    #             -2.0
    #             * PI
    #             * self.TDI_data.frequency_samples[i]
    #             * self.orbit.arm_length_sec
    #             * ComplexNumber([0, 1])
    #         )

    #         prefactor = ComplexNumber(0.0, 0.0)
    #         if ti.static(self.TDI_data.data_info.generation == "1.5"):
    #             prefactor = ComplexNumber(1, 0) - tm.cpow(z, 2)
    #         elif ti.static(self.TDI_data.data_info.generation == "2.0"):
    #             prefactor = (
    #                 ComplexNumber(1, 0) - tm.cpow(z, 2) - tm.cpow(z, 4) + tm.cpow(z, 6)
    #             )
    #         self._FD_response_assistance[i]["TDI_generation_prefactor"] = prefactor
    #         self._FD_response_assistance[i]["delay_factor"] = z

    # @ti.kernel
    # def update_frequency_domain_response(
    #     self, waveform: ti.template(), lam: ti.f64, beta: ti.f64, psi: ti.f64
    # ):
    #     """
    #     TODO: move the main calculation into tdi class maybe better?
    #     keep `waveform` point to the same memory address to avoid kernal repeated instantiation
    #     the computaion is evaluated in the order of frequency point
    #     using AoS structure to store data for efficiency
    #     note:
    #     n1: link3->2
    #     n2: link1->3
    #     n3: link2->1
    #     """
    #     pol_tensor = polarization_tensor_SSB(lam, beta, psi)  # matrix: 3*3
    #     k = GW_propagation_unit_vector(lam, beta)  # vector: 3

    #     for i in self.response_container:
    #         constellation_vectors = self.orbit.orbit_vectors(waveform[i].tf)

    #         n1_h_n1 = (
    #             constellation_vectors.n1
    #             @ pol_tensor.plus
    #             @ constellation_vectors.n1
    #             * hp
    #             + constellation_vectors.n1
    #             @ pol_tensor.cross
    #             @ constellation_vectors.n1
    #             * hc
    #         )  # complex number
    #         n2_h_n2 = (
    #             constellation_vectors.n2
    #             @ pol_tensor.plus
    #             @ constellation_vectors.n2
    #             * hp
    #             + constellation_vectors.n2
    #             @ pol_tensor.cross
    #             @ constellation_vectors.n2
    #             * hc
    #         )  # complex number
    #         n3_h_n3 = (
    #             constellation_vectors.n3
    #             @ pol_tensor.plus
    #             @ constellation_vectors.n3
    #             * hp
    #             + constellation_vectors.n3
    #             @ pol_tensor.cross
    #             @ constellation_vectors.n3
    #             * hc
    #         )  # complex number

    #         k_n1 = k @ constellation_vectors.n1  # scalar
    #         k_n2 = k @ constellation_vectors.n2  # scalar
    #         k_n3 = k @ constellation_vectors.n3  # scalar

    #         k_p1_p2 = k @ (
    #             constellation_vectors.p1 + constellation_vectors.p2
    #         )  # scalar
    #         k_p2_p3 = k @ (
    #             constellation_vectors.p2 + constellation_vectors.p3
    #         )  # scalar
    #         k_p3_p1 = k @ (
    #             constellation_vectors.p3 + constellation_vectors.p1
    #         )  # scalar

    #         common_sinc = (
    #             PI * self.TDI_data.frequency_samples[i] * self.orbit.arm_length_sec
    #         )  # scalar
    #         sinc21 = sinc(common_sinc * (1.0 - k_n3))  # scalar
    #         sinc12 = sinc(common_sinc * (1.0 + k_n3))  # scalar
    #         sinc32 = sinc(common_sinc * (1.0 - k_n1))  # scalar
    #         sinc23 = sinc(common_sinc * (1.0 + k_n1))  # scalar
    #         sinc13 = sinc(common_sinc * (1.0 - k_n2))  # scalar
    #         sinc31 = sinc(common_sinc * (1.0 + k_n2))  # scalar

    #         common_exp = (
    #             -PI * self.TDI_data.frequency_samples[i] * ComplexNumber([0.0, 1.0])
    #         )  # ComplexNumber
    #         exp12 = tm.cexp(
    #             common_exp * (self.orbit.arm_length_sec + k_p1_p2)
    #         )  # ComplexNumber
    #         exp23 = tm.cexp(
    #             common_exp * (self.orbit.arm_length_sec + k_p2_p3)
    #         )  # ComplexNumber
    #         exp31 = tm.cexp(
    #             common_exp * (self.orbit.arm_length_sec + k_p3_p1)
    #         )  # ComplexNumber

    #         prefactor = (
    #             -PI
    #             * self.TDI_data.frequency_samples[i]
    #             * self.orbit.arm_length_sec
    #             * ComplexNumber([0.0, 1.0])
    #         )  # ComplexNumber

    #         self._FD_response_assistance[i]["single_links"]["link12"] = (
    #             sinc12 * tm.cmul(tm.cmul(prefactor, n3_h_n3), exp12)
    #         )  # ComplexNumber
    #         self._FD_response_assistance[i]["single_links"]["link21"] = (
    #             sinc21 * tm.cmul(tm.cmul(prefactor, n3_h_n3), exp12)
    #         )  # ComplexNumber
    #         self._FD_response_assistance[i]["single_links"]["link23"] = (
    #             sinc23 * tm.cmul(tm.cmul(prefactor, n1_h_n1), exp23)
    #         )  # ComplexNumber
    #         self._FD_response_assistance[i]["single_links"]["link32"] = (
    #             sinc32 * tm.cmul(tm.cmul(prefactor, n1_h_n1), exp23)
    #         )  # ComplexNumber
    #         self._FD_response_assistance[i]["single_links"]["link31"] = (
    #             sinc31 * tm.cmul(tm.cmul(prefactor, n2_h_n2), exp31)
    #         )  # ComplexNumber
    #         self._FD_response_assistance[i]["single_links"]["link13"] = (
    #             sinc13 * tm.cmul(tm.cmul(prefactor, n2_h_n2), exp31)
    #         )  # ComplexNumber

    #         for chan in ti.static(self.TDI_data.data_info.channels):
    #             self.response_container[i][chan] = tm.cmul(
    #                 self._FD_response_assistance[i]["TDI_generation_prefactor"],
    #                 TDI_combine_function_FD[chan](
    #                     self._FD_response_assistance[i]["delay_factor"],
    #                     self._FD_response_assistance[i]["single_links"],
    #                 ),
    #             )

    # def inject_fd_signal(
    #     self,
    #     waveform: ti.StructField | dict[str, NDArray[np.complex128]],
    #     lam: ti.f64,
    #     beta: ti.f64,
    #     psi: ti.f64,
    # ) -> None:
    #     """
    #     TODO: Using dict[NDArray] as input requires instantiate a new waveform_field in each
    #     function call, which can lead to repeated instantiation of the kernel for
    #      the  `updata_frequency_domain_response` method, and deteriorate computaional
    #      efficiency. If there are many signals to inject, using a ti.StructField with
    #      the same memroy address as the input.
    #      This
    #     """
    #     if isinstance(waveform, ti.StructField):
    #         if not waveform.shape == (self.TDI_data.data_info.frequency_series_length,):
    #             raise ValueError(
    #                 "Cannot perfrom injection, since the shape of input \
    #                              waveform is different with the TDI data"
    #             )
    #         waveform_field = waveform

    #     elif isinstance(waveform, dict):
    #         if not all(
    #             [
    #                 len(data) == self.TDI_data.data_info.frequency_series_length
    #                 for _, data in waveform.items()
    #             ]
    #         ):
    #             raise ValueError(
    #                 "Cannot perfrom injection, since there is at least one \
    #                              array in the input dict having different length with \
    #                              the TDI data."
    #             )
    #         waveform_field = ti.Struct.field(
    #             dict.fromkeys(waveform.keys(), ComplexNumber),
    #             shape=(self.TDI_data.data_info.frequency_series_length,),
    #         )
    #         complex_numpy_array_dict_to_taichi_field(waveform, waveform_field)
    #     else:
    #         raise TypeError(
    #             "Unsupported type, expect `ti.StructField` or `dict[NDArray]`"
    #         )

    #     self.update_frequency_domain_response(waveform_field, lam, beta, psi)
    #     self.TDI_data.add_into_frequency_domian_data(self.response_container)


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
        )

    @ti.kernel
    def update_single_link_response(
        self,
        waveform: ti.template(),
        lam: ti.f64,
        beta: ti.f64,
        psi: ti.f64,
        tc: ti.f64,
    ):
        pol_tensor = get_polarization_tensor_ssb(lam, beta, psi)  # matrix: 3*3
        k = get_gw_propagation_unit_vector(lam, beta)  # vector: 3

        for i in self.detector.single_link_response:
            fi = self.detector.tdi_data.frequency_samples[i]
            cexp_tshift = tm.cexp(ComplexNumber([0.0, -2.0 * PI * fi * tc]))
            hp = tm.cmul(waveform[i].plus, cexp_tshift)
            hc = tm.cmul(waveform[i].cross, cexp_tshift)
            tf = waveform[i].tf + tc
            constellation_vectors = self.detector.orbit_model.get_constellation_vectors(tf)  # fmt: skip

            # n1: unit vector of 2 -> 3
            n1_h_n1 = (
                constellation_vectors.n1
                @ pol_tensor.plus
                @ constellation_vectors.n1
                * hp
                + constellation_vectors.n1
                @ pol_tensor.cross
                @ constellation_vectors.n1
                * hc
            )  # complex number
            # n2: unit vector of 3 -> 1
            n2_h_n2 = (
                constellation_vectors.n2
                @ pol_tensor.plus
                @ constellation_vectors.n2
                * hp
                + constellation_vectors.n2
                @ pol_tensor.cross
                @ constellation_vectors.n2
                * hc
            )  # complex number
            # n3: unit vector of 1 -> 2
            n3_h_n3 = (
                constellation_vectors.n3
                @ pol_tensor.plus
                @ constellation_vectors.n3
                * hp
                + constellation_vectors.n3
                @ pol_tensor.cross
                @ constellation_vectors.n3
                * hc
            )  # complex number

            k_n1 = k @ constellation_vectors.n1  # scalar
            k_n2 = k @ constellation_vectors.n2  # scalar
            k_n3 = k @ constellation_vectors.n3  # scalar

            k_x1_x2 = k @ (
                constellation_vectors.x1 + constellation_vectors.x2
            )  # scalar
            k_x2_x3 = k @ (
                constellation_vectors.x2 + constellation_vectors.x3
            )  # scalar
            k_x3_x1 = k @ (
                constellation_vectors.x3 + constellation_vectors.x1
            )  # scalar

            pi_f_L = PI * fi * self.detector.orbit_model.arm_length_sec  # scalar
            sinc32 = sinc(pi_f_L * (1.0 - k_n1))  # scalar
            sinc23 = sinc(pi_f_L * (1.0 + k_n1))  # scalar
            sinc13 = sinc(pi_f_L * (1.0 - k_n2))  # scalar
            sinc31 = sinc(pi_f_L * (1.0 + k_n2))  # scalar
            sinc21 = sinc(pi_f_L * (1.0 - k_n3))  # scalar
            sinc12 = sinc(pi_f_L * (1.0 + k_n3))  # scalar

            common_exp = -PI * fi * ComplexNumber([0.0, 1.0])  # ComplexNumber
            exp12 = tm.cexp(
                common_exp * (self.detector.orbit_model.arm_length_sec + k_x1_x2)
            )  # ComplexNumber
            exp23 = tm.cexp(
                common_exp * (self.detector.orbit_model.arm_length_sec + k_x2_x3)
            )  # ComplexNumber
            exp31 = tm.cexp(
                common_exp * (self.detector.orbit_model.arm_length_sec + k_x3_x1)
            )  # ComplexNumber

            prefactor = -pi_f_L * ComplexNumber([0.0, 1.0])  # ComplexNumber

            self.detector.single_link_response[i].link12 = sinc12 * tm.cmul(
                tm.cmul(prefactor, n3_h_n3), exp12
            )  # ComplexNumber
            self.detector.single_link_response[i].link21 = sinc21 * tm.cmul(
                tm.cmul(prefactor, n3_h_n3), exp12
            )  # ComplexNumber
            self.detector.single_link_response[i].link23 = sinc23 * tm.cmul(
                tm.cmul(prefactor, n1_h_n1), exp23
            )  # ComplexNumber
            self.detector.single_link_response[i].link32 = sinc32 * tm.cmul(
                tm.cmul(prefactor, n1_h_n1), exp23
            )  # ComplexNumber
            self.detector.single_link_response[i].link31 = sinc31 * tm.cmul(
                tm.cmul(prefactor, n2_h_n2), exp31
            )  # ComplexNumber
            self.detector.single_link_response[i].link13 = sinc13 * tm.cmul(
                tm.cmul(prefactor, n2_h_n2), exp31
            )  # ComplexNumber


class FDResponseModelLongWavelength(SingleLinkResponseModel):
    pass


class FDResponseModelStaticLongWavelength(SingleLinkResponseModel):
    pass


# @ti.data_oriented
# class TDResponseModelConstantEqualArmCornish2003(SingleLinkResponseModel):

#     def __init__(self, interpolate_kernel: str | tuple[str, int]):
#         if isinstance(interpolate_kernel, str) and (interpolate_kernel == "linear"):
#             self.interpolate_kernel = linear_interpolate
#             self.interpolate_kernle_length = 3
#         elif isinstance(interpolate_kernel, tuple):
#             self.interpolate_kernel = None
#             self.interpolate_kernel_length = interpolate_kernel[1]

#     def init_single_link_response_model(self, detector: InterferometerAntenna) -> None:
#         self.detector = weakref.proxy(detector)
#         self.detector.single_link_response = SingleLinkStructReal.field(
#             shape=(self.detector.tdi_combination.extended_time_series_length,),
#         )

#         self.extended_time_samples = ti.field(
#             ti.f64, shape=(self.detector.tdi_combination.extended_time_series_length,)
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
#         self, waveform: ti.types.ndarray(dtype=ti.f64, ndim=2), time: ti.f64
#     ):
#         dt = self.detector.tdi_data.data_info.delta_time
#         idx = time // dt
#         frac = time % dt
#         hp_left, hc_left = waveform[idx, 0], waveform[idx, 1]
#         hp_right, hc_right = waveform[idx + 1, 0], waveform[idx + 1, 1]
#         hp = linear_interpolate(hp_left, hp_right, frac)
#         hc = linear_interpolate(hc_left, hc_right, frac)
#         return hp, hc

#     def _ensure_waveform_length(self, waveform_container:dict[str, NDArray[np.float64] | float],
#                                 tc:ti.f64):
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
#         waveform_container: dict[str, NDArray[np.float64] | float],
#         lam: ti.f64,
#         beta: ti.f64,
#         psi: ti.f64,
#         tc: ti.f64,
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
#         waveform: ti.types.ndarray(dtype=ti.f64, ndim=2),
#         t0: ti.f64,  # time of the first data point in waveform
#         lam: ti.f64,
#         beta: ti.f64,
#         psi: ti.f64,
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
# old implementaion
#     def initialize_TDI_data(self):
#         '''
#         set TDI_data field, using AoS structure to store data for efficiency,
#         keep the memory address fixed to avoid repeated repeated instantiation of the computational kernel
#         {frequencies: ti.f64,
#          delay_factor: ComplexNumber,
#          TDI_gen_prefactor: ComplexNumber,
#          single_links: SingleLinksStruct,
#          channels_data: ti.types.struct(TDI_chan_dict)
#          }
#         '''
#         TDI_chan_dict = dict.fromkeys(self.TDI_channels, ComplexNumber)
#         TDI_chan_struct = ti.types.struct(**TDI_chan_dict)

#         TDI_data_struct = ti.types.struct(frequencies = ti.f64,
#                                           delay_factor = ComplexNumber,
#                                           TDI_gen_prefactor = ComplexNumber,
#                                           single_links = SingleLinksStruct,
#                                           channels_data = TDI_chan_struct)
#         TDI_data_field = TDI_data_struct.field()
#         ti.root.dense(ti.i, self.data_length).place(TDI_data_field)

#         # set frequencies field
#         TDI_data_field.frequencies.copy_from(self.frequencies)
#         # set dalay_factor and TDI_gen_prefactor
#         if self.TDI_generation == '1.5':
#             int_TDI_gen = 1
#         elif self.TDI_generation == '2.0':
#             int_TDI_gen = 2
#         _compute_TDI_prefactor(TDI_data_field.frequencies, TDI_data_field.delay_factor, TDI_data_field.TDI_gen_prefactor,
#                                self.armlength_sec, int_TDI_gen)

#         self.TDI_data = TDI_data_field

#         return None


#     def initialize_waveform_container(self):
#         waveform_field = ti.Struct.field({'plus': ComplexNumber,
#                                           'cross': ComplexNumber,
#                                           'tf': ti.f64})
#         ti.root.dense(ti.i, self.data_length).place(waveform_field)
#         self.waveform_container = waveform_field
#         return None


#     def updata_TDI_responses(self, parameters):
#         '''
#         compute the strain of TDI channels from given waveform

#         Parameters
#         ==========
#         waveform: dict
#             contains the keys "amplitude", "phase", "tf", "frequencies"
#         parameters: dict
#             parameters describes the GW source

#         Returns:
#         ========
#         dict, strains of TDI channels of current instance
#         '''
#         _generate_TDI_responses(self.TDI_data, self.waveform_container, self._orbit_vectors_func, self.armlength_sec,
#                                 parameters['ecliptic_longitude'],  parameters['ecliptic_latitude'],  parameters['polarization'])
#         return None


#     def initialize_strains_FD(self):
#         strains_FD_field = ti.Struct.field(dict.fromkeys(self.TDI_channels, ComplexNumber))
#         ti.root.dense(ti.i, self.data_length).place(strains_FD_field)
#         self.strains_FD = strains_FD_field
#         return None


#     def initialize_strains_TD(self):
#         self.strains_TD = None
#         return None


#     def inject_signal_FD(self, parameters, waveform):
#         '''
#         TODO wavefrom_dictionary
#         inject the GW signal into the detector strains

#         Parameters
#         ==========
#         parameters: dict
#             parameters describes the GW source
#         waveform: waveform object which contains the detector.waveform_container

#         '''
#         waveform.update_waveform(parameters)
#         self.updata_TDI_responses(parameters)
#         _inject_into_strains_FD(self.strains_FD, self.TDI_data.channels_data)

#         injected_signals = ti.Struct.field(dict.fromkeys(self.TDI_channels, ComplexNumber), shape=(self.data_length,))
#         injected_signals.copy_from(self.TDI_data.channels_data)
#         self.signals.append(injected_signals)

#         return None

#     def initialize_PSDs(self):
#         self.PSDs = ti.Struct.field(dict.fromkeys(self.TDI_channels, ti.f64), shape=(self.data_length,))
#         return None

#     def set_PSDs_from_noise_model(self):
#         '''
#         compute the psd array from the give noise model

#         Parameters
#         ==========
#         frequencies: array,
#             default is None which will use the self.frequencies

#         Returns:
#         ========
#         dict, psd array of each TDI channels
#         '''
#         PSDs_array = {}
#         for chan in self.TDI_channels:
#             PSDs_array[chan] = noise_models[self.psd_model](self._np_array_frequenices, chan, self.TDI_generation)
#         self.PSDs.from_numpy(PSDs_array)
#         self._np_array_PSDs = PSDs_array
#         return None

#     @property
#     def np_array_PSDs(self):
#         return self._np_array_PSDs


#     def inject_noise_FD_realization_from_psd(self, seed=None):
#         '''
#         generate a noise realization from psd
#         (eq.12) in https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.023033

#         Parameters
#         ==========
#         seed: integer,
#             set the seed for predictable random number sequence, default is None
#         '''
#         rng = np.random.default_rng(seed=seed)
#         var = 0.5 * (1. / self.delta_f)**0.5
#         noise_strains = {}
#         for chan in self.TDI_channels:
#             # noise_amp = rng.normal(0, var, num) * (self.psd_array[chan])**0.5
#             # random_phase = rng.uniform(0, 2*PI, num)
#             # noise_chan = noise_amp * np.exp(1j*random_phase)
#             re = rng.normal(0, var, self.data_length) * (self._np_array_PSDs[chan])**0.5
#             im = rng.normal(0, var, self.data_length) * (self._np_array_PSDs[chan])**0.5
#             noise_strains[chan] = np.vstack((re, im)).T

#         noise_strains_field = ti.Struct.field(dict.fromkeys(self.TDI_channels, ComplexNumber), shape=(self.data_length, ))
#         noise_strains_field.from_numpy(noise_strains)
#         _inject_into_strains_FD(self.strains_FD, noise_strains_field)

#         return None


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


# def plot_FD_data_amplitude(self, outdir='.', contents=['strains_FD', 'signals', 'noise', 'psd_array']):
#     '''
#     plot the FD data in the instance

#     Parameters
#     ==========
#     outdir: string
#         outdit for saving the figure
#     contents: list
#         contents in the figure, all available contents are ['strains_FD', 'signals', 'noise', 'psd_array']
#     '''
#     if ('signals' in contents) and (len(self.signals)==0):
#         print(f'Warning: You are requiring to plot `signals`, which do not contain any injections and will be neglicted, call `inject_signal_FD` first.')
#         contents.remove('signals')
#     for item in contents[:]:    # using the copy of the list to avoid the unexpected result
#         if getattr(self, item) is None:
#             print(f'Warning: You are requiring to plot {item}, which do not contained in your detector instance and will be neglicted.')
#             contents.remove(item)

#     for chan in self.TDI_channels:
#         fig, ax = plt.subplots()
#         ax.set_title(f'channel {chan}; generation {self.TDI_generation}')
#         if 'noise' in contents:
#             ax.loglog(self.frequencies, np.abs(self.noise[chan]), color='C2', label='noise realization')
#         if 'strains_FD' in contents:
#             ax.loglog(self.frequencies, np.abs(self.strains_FD[chan]), color='C0', label='total strain')
#         if 'signals' in contents:
#             for idx, injection in enumerate(self.signals):
#                 ax.loglog(self.frequencies, np.abs(injection[chan]), color='C1', label=f'injected GW signal {idx}')
#         if 'psd_array' in contents:
#             ax.loglog(self.frequencies, 0.5*np.sqrt(self.psd_array[chan])*(self.duration)**0.5, color='C3', label=r'$\frac{1}{2}\sqrt{S_n(f)T}$')

#         ax.grid(True)
#         ax.set_ylabel(r'Strain $[1/{\rm Hz}]$')
#         ax.set_xlabel(r'Frequency [Hz]')
#         ax.legend(loc='best')
#         fig.tight_layout()
#         fig.savefig('{}/{}_{}{}_data_FD.png'.format(outdir, self.name, chan, self.TDI_generation))
#         plt.close(fig)

#     return None


# def plot_TD_data(self, contents=['signal', 'noise']):
#     pass
#     return None


# def save_detector_data(self, outdir='.', label=None):
#     '''
#     TODO save the parameters of injected signals
#     save the data in the instance to a hdf5 file, save contents: [signals, noise, strains_FD, strains_TD,
#     frequencies, psd_array]

#     Parameters
#     ==========
#     outdir: string
#     '''
#     contents = ['signals', 'noise', 'strains_FD', 'strains_TD', 'frequencies', 'psd_array']
#     save_dict = {}
#     for item in contents:
#         save_dict[item] = getattr(self, item)

#     filename = f'{outdir}/{self.name}_detector_data_{label}.hdf5'
#     with h5py.File(filename, 'w') as file:
#         recursively_save_dict_contents_to_group(file, '/', save_dict)

#     return None

# def set_detector_data_from_file(self, filename):
#     '''
#     uncompleted !!!
#     set ['signals', 'noise', 'strains_FD', 'strains_TD', 'frequencies', 'psd_array']
#     TODO this function is incomplete !!! when use this func the frequencies, psd_array, ... may not have the same shape
#     TODO consider the conflict with read-in data and already set data
#     TODO add supportation of other attribute,
#     set the strains_FD from h5py file

#     Parameters
#     ==========
#     filename: string
#         hdf5 file containing the 'strains_FD'
#     '''
#     with h5py.File(filename, 'r') as file:
#         data = recursively_load_dict_contents_from_group(file, '/')
#     # TODO corresponding frequencies, duration, delta_time should be check

#     self.frequencies = data['frequencies']
#     self.psd_array =  data['psd_array']
#     self.strains_TD = data['strains_TD']
#     self.strains_FD = data['strains_FD']
#     self.noise = data['noise']
#     self.signals = list(data['signals'].values())

#     return None

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tdi import TDIChannelData, TDICombinationModel
