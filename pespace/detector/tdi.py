"""Module for handling TDI combinations and data of TDI channels."""
# from __future__ import annotations
# since the type hint in current taichi-lang does not support to parse types from strings,
# use string literal types for foward reference in python scope.
import warnings
from dataclasses import dataclass, field, asdict
import weakref
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy import signal
import h5py
import taichi as ti
import taichi.math as tm

from .noise import FrequencyDomainNoiseModel, available_noise_models
from ..utils.utils import (
    linear_interpolate_kernel,
    complex_numpy_array_dict_to_taichi_field,
    taichi_field_to_complex_numpy_array_dict,
    ti_complex,
    SingleLinkStructReal,
)
from ..utils.constants import *


# @ti.func
# def _TDI_A_FD(
#     z: ti_complex, singlelink_responses: SingleLinksStruct
# ) -> ti_complex:
#     """
#     Function for computing A channel of TDI combination in frequency domain.

#     Parameters:
#     ===========
#     z:
#         Delay factor, exp(-1j*2*PI*f*arm_length_sec).
#     singlelink_responses:
#         Responses of each link.

#     Returns:
#     ========
#     A channel without the generation prefactor.
#     """
#     return (
#         singlelink_responses["link23"]
#         + tm.cmul(z, singlelink_responses["link32"])
#         + singlelink_responses["link21"]
#         + tm.cmul(z, singlelink_responses["link12"])
#         - tm.cmul(
#             (ti_complex(1, 0) + z),
#             (singlelink_responses["link13"]) + singlelink_responses["link31"],
#         )
#     ) / tm.sqrt(2)


# @ti.func
# def _TDI_E_FD(
#     z: ti_complex, singlelink_responses: SingleLinksStruct
# ) -> ti_complex:
#     """
#     Function for computing E channel of TDI combination in frequency domain.

#     Parameters:
#     ===========
#     z:
#         Delay factor, exp(-1j*2*PI*f*arm_length_sec).
#     singlelink_responses:
#         Responses of each link.

#     Returns:
#     ========
#     E channel without the generation prefactor.
#     """
#     return (
#         tm.cmul(
#             (ti_complex(1, 0) - z),
#             (singlelink_responses["link31"] - singlelink_responses["link13"]),
#         )
#         + tm.cmul(
#             (z + ti_complex(2, 0)),
#             (singlelink_responses["link32"] - singlelink_responses["link12"]),
#         )
#         + tm.cmul(
#             (ti_complex(1, 0) + 2 * z),
#             (singlelink_responses["link23"] - singlelink_responses["link21"]),
#         )
#     ) / tm.sqrt(6)


# @ti.func
# def _TDI_T_FD(
#     z: ti_complex, singlelink_responses: SingleLinksStruct
# ) -> ti_complex:
#     """
#     Function for computing T channel of TDI combination in frequency domain.

#     Parameters:
#     ===========
#     z:
#         Delay factor, exp(-1j*2*PI*f*arm_length_sec).
#     singlelink_responses:
#         Responses of each link.

#     Returns:
#     ========
#     T channel without the generation prefactor.
#     """
#     return (
#         tm.cmul(
#             (
#                 singlelink_responses["link12"]
#                 - singlelink_responses["link21"]
#                 + singlelink_responses["link23"]
#                 - singlelink_responses["link32"]
#                 + singlelink_responses["link31"]
#                 - singlelink_responses["link13"]
#             ),
#             (ti_complex(1, 0) - z),
#         )
#     ) / tm.sqrt(3)


@ti.kernel
def _add_input_field_into_tdi_data(tdi_data: ti.template(), input: ti.template()):
    """
    Add the input into TDI data.

    The input field must have the same shape and channels as ``TDI_data``.
    ``StructField`` has the default layout of AoS, unrolling the channels in the inner 
    loop for memory accessing efficiency. This function is dimensionality-independent 
    and can be used for TDI data of all domains.

    Parameters
    ----------
    tdi_data : ti.template()
        Target TDI data field to add into.
    input : ti.template()
        Input field to be added, must have same shape and channels as tdi_data.
    """
    for I in ti.grouped(tdi_data):  # dimensionality independent
        for chan in ti.static(tdi_data.keys):  # beter performance for AOS layout
            tdi_data[I][chan] += input[I][chan]


@dataclass(frozen=True)
class DataInformation:
    """
    Store TDI channel data information.

    Attributes
    ----------
    channels : tuple[str, ...]
        TDI channel names.
    duration : float
        Observing duration of data, in seconds.
    delta_time : float
        Sampling cadence, in seconds.
    start_time : float
        Time label for the first time sample, in seconds.
    minimum_frequency : float
        Minimum frequency for the limited frequency band, in Hz.
    maximum_frequency : float
        Maximum frequency for the limited frequency band, in Hz.
    sampling_frequency : float
        Sampling frequency, computed as 1/delta_time.
    delta_frequency : float
        Frequency resolution, computed as 1/duration.
    time_series_length : int
        Number of time samples.
    time_samples_array : NDArray
        Numpy array of time sample values.
    full_frequency_series_length : int
        Length of full frequency series from FFT.
    full_frequency_samples_array : NDArray
        Array of all frequency samples from FFT.
    frequency_mask_array : NDArray[np.bool_]
        Boolean mask for selecting frequencies within [minimum_frequency, maximum_frequency].
    frequency_samples_array : NDArray
        Numpy array of frequency samples within the specified range.
    frequency_series_length : int
        Number of frequency samples within the specified range.
    """

    channels: tuple[str, ...]
    duration: float
    delta_time: float
    start_time: float
    minimum_frequency: float
    maximum_frequency: float

    sampling_frequency: float = field(init=False)
    delta_frequency: float = field(init=False)
    time_series_length: int = field(init=False)
    time_samples_array: NDArray = field(init=False)
    full_frequency_series_length: int = field(init=False)
    full_frequency_samples_array: NDArray = field(init=False)
    frequency_mask_array: NDArray[np.bool_] = field(init=False)
    frequency_samples_array: NDArray = field(init=False)
    frequency_series_length: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Generate derived quantities from duration and delta_time.

        Computes sampling frequencies, array lengths, and time/frequency sample arrays.
        Sets proper time and frequency samples based on the input parameters.
        Time samples are generated as: start_time + n * delta_time for n = 0, 1, ..., time_series_length-1
        Frequency samples are generated using numpy.fft.rfftfreq and then masked to the specified range.
        """
        sampling_frequency = 1 / self.delta_time
        delta_frequency = 1 / self.duration
        time_series_length = int(self.duration // self.delta_time + 1)
        time_samples_array = (
            np.arange(time_series_length) * self.delta_time + self.start_time
        )

        full_frequency_samples_array = np.fft.rfftfreq(
            time_series_length, self.delta_time
        )
        full_frequency_series_length = int(len(full_frequency_samples_array))

        frequency_mask_array = (
            full_frequency_samples_array >= self.minimum_frequency
        ) * (full_frequency_samples_array <= self.maximum_frequency)
        frequency_samples_array = full_frequency_samples_array[frequency_mask_array]
        frequency_series_length = int(len(frequency_samples_array))

        object.__setattr__(self, "sampling_frequency", sampling_frequency)
        object.__setattr__(self, "delta_frequency", delta_frequency)
        object.__setattr__(self, "time_series_length", time_series_length)
        object.__setattr__(self, "time_samples_array", time_samples_array)
        object.__setattr__(
            self, "full_frequency_series_length", full_frequency_series_length
        )
        object.__setattr__(
            self, "full_frequency_samples_array", full_frequency_samples_array
        )
        object.__setattr__(self, "frequency_mask_array", frequency_mask_array)
        object.__setattr__(self, "frequency_samples_array", frequency_samples_array)
        object.__setattr__(self, "frequency_series_length", frequency_series_length)


class TDIChannelData:
    """Store and manipulate TDI channel data."""

    def __init__(self, scaling: bool = False) -> None:
        """
        Initialize TDIChannelData instance.

        Parameters
        ----------
        scaling : bool, optional
            Whether to apply scaling to the data. Default is False.
            If False and using ``f32`` precision, a warning will be issued about
            potential numerical errors.

        Raises
        ------
        UserWarning
            If scaling is False and default float precision is ``f32``.
        """
        self.scaling = scaling
        if (not scaling) and (
            ti.lang.impl.current_cfg().default_fp.to_string() == "f32"
        ):
            warnings.warn(
                "The current default float precision is f32, but no scaling has been applied. "
                "Please be aware of potential numerical errors or underflow issues. "
            )

        if ti.lang.impl.current_cfg().default_fp.to_string() == "f32":
            self._np_fp = np.float32
            self._np_cp = np.complex64
        else:
            self._np_fp = np.float64
            self._np_cp = np.complex128

        self._initialize_state()

    def _reset(self) -> None:
        """Reset the instance to initial state."""
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize all data fields to None."""
        self.time_samples = None
        self.frequency_samples = None
        self.wavelet_samples = None

        self.td_data = None
        self.fd_data = None
        self.wd_data = None

        # self.td_noise_correlation_function = None
        self.fd_noise_power_density = None
        self.wd_noise_power_density = None

        self._data_info = None
        self._reset_flag = False

    @property
    def data_info(self) -> None | DataInformation:
        """
        Get data information.

        Returns
        -------
        DataInformation or None
            Metadata about the TDI data, or None if not set.
        """
        return self._data_info

    def set_data_info(
        self,
        channels: tuple[str, ...],
        duration: float,
        delta_time: float,
        start_time: float,
        minimum_frequency: float,
        maximum_frequency: float,
    ) -> None:
        """
        Set data information for TDI channels.

        Parameters
        ----------
        channels : tuple[str, ...]
            TDI channel names.
        duration : float
            Observing duration of data, in seconds.
        delta_time : float
            Sampling cadence, in seconds.
        start_time : float
            Time label for the first time sample, in seconds.
        minimum_frequency : float
            Minimum frequency for the limited frequency band, in Hz.
        maximum_frequency : float
            Maximum frequency for the limited frequency band, in Hz.

        Raises
        ------
        UserWarning
            If ``data_info`` is being set when TDI data already exists.
        """
        if self._reset_flag:
            warnings.warn(
                "You are setting `data_info`, whereas you have set TDI data of current "
                "instance previously. Setting `data_info` along may lead mismatch of "
                "`data_info` and the stored data. Please check whether this is intentional."
            )
        self._data_info = DataInformation(
            channels,
            duration,
            delta_time,
            start_time,
            minimum_frequency,
            maximum_frequency,
        )

    def _init_td_data(self) -> None:
        """
        Initialize ``taichi.field`` for time domain data.

        Creates ``time_samples`` and ``td_data`` fields. This is an internal method
        that should be called after setting ``data_info``. Use ``set_td_data_from_zero``
        or ``set_td_data_from_input`` to set time domain data externally.
        """
        self.time_samples = ti.field(float, (self.data_info.time_series_length,))
        self.time_samples.from_numpy(
            np.astype(self.data_info.time_samples_array, self._np_fp)
        )
        self.td_data = ti.Struct.field(
            dict.fromkeys(self.data_info.channels, float),
            shape=(self.data_info.time_series_length,),
        )

    def _init_fd_data(self) -> None:
        """
        Initialize ``taichi.field`` for frequency domain data.

        Creates ``frequency_samples`` and ``fd_data`` fields. This is an internal method
        that should be called after setting ``data_info``. Use ``set_fd_data_from_zero``
        or ``set_fd_data_from_input`` to set frequency domain data externally.
        """
        self.frequency_samples = ti.field(
            float, (self.data_info.frequency_series_length,)
        )
        self.frequency_samples.from_numpy(
            np.astype(self.data_info.frequency_samples_array, self._np_fp)
        )
        self.fd_data = ti.Struct.field(
            dict.fromkeys(self.data_info.channels, ti_complex),
            shape=(self.data_info.frequency_series_length,),
        )

    def _init_wd_data(self) -> None:
        """
        Initialize ``taichi.field`` for wavelet domain data.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def set_td_data_from_input(
        self,
        channels: tuple[str, ...],
        duration: float,
        delta_time: float,
        tdi_data_array: NDArray,
        start_time: float = 0.0,
        minimum_frequency: float = 1e-5,
        maximum_frequency: float = 0.1,
    ) -> None:
        """
        Set time domain TDI data from input numpy array.

        Parameters
        ----------
        channels : tuple[str, ...]
            TDI channel names.
        duration : float
            Observing duration of data, in seconds.
        delta_time : float
            Sampling cadence, in seconds.
        tdi_data_array : NDArray
            Array storing TDI data with shape ``(len(channels), time_series_length)``.
            The order in channels list must match the order in ``tdi_data_array``.
        start_time : float, optional
            Time label for the first time sample, in seconds. Default is 0.0.
        minimum_frequency : float, optional
            Minimum frequency for the limited frequency band, in Hz. Default is 1e-5.
        maximum_frequency : float, optional
            Maximum frequency for the limited frequency band, in Hz. Default is 0.1.

        Raises
        ------
        ValueError
            If the shape of `tdi_data_array` does not match the expected dimensions.
        UserWarning
            If TDI data has been set previously, the instance will be reset.
        """
        if self._reset_flag:
            warnings.warn(
                "You are setting `td_data` with input array, whereas you have set TDI "
                "data of current instance previously. Please check whether this is intertional.\n"
                "In order to avoid potential errors, current instance is reset. Please "
                "regenerate TDI data of other domian or noise behavior data if needed."
            )
            self._reset()

        self.set_data_info(
            channels,
            duration,
            delta_time,
            start_time,
            minimum_frequency,
            maximum_frequency,
        )
        channels_num, samples_num = tdi_data_array.shape
        if not len(channels) == channels_num:
            raise ValueError(
                f"You set channenls with {channels}, while the length of first dimension "
                f"of input tdi_data_array is {channels_num}."
            )
        if not self.data_info.time_series_length == samples_num:
            raise ValueError(
                f"The length of second dimension of input array is {samples_num}. It is "
                f"different with the `time_series_length={self.data_info.time_series_length}` "
                f"which is set according to the duration and delta_time by `int(duration//delta_time + 1)`."
            )

        self._init_td_data()
        self.td_data.from_numpy(
            dict(zip(channels, np.astype(tdi_data_array, self._np_fp)))
        )

        self._reset_flag = True

    def set_fd_data_from_input(
        self,
        channels: tuple[str, ...],
        duration: float,
        delta_time: float,
        tdi_data_array: NDArray,
        start_time: float = 0.0,
        minimum_frequency: float = 1e-5,
        maximum_frequency: float = 0.1,
    ) -> None:
        """
        Set frequency domain TDI data from input numpy array.

        Parameters
        ----------
        channels : tuple[str, ...]
            TDI channel names.
        duration : float
            Observing duration of data, in seconds.
        delta_time : float
            Sampling cadence, in seconds.
        tdi_data_array : NDArray
            Complex array storing TDI data with shape ``(len(channels), frequency_series_length)``.
            The order in channels list must match the order in ``tdi_data_array``.
        start_time : float, optional
            Time label for the first time sample, in seconds. Default is 0.0.
        minimum_frequency : float, optional
            Minimum frequency for the limited frequency band, in Hz. Default is 1e-5.
        maximum_frequency : float, optional
            Maximum frequency for the limited frequency band, in Hz. Default is 0.1.

        Raises
        ------
        ValueError
            If the shape of ``tdi_data_array`` does not match the expected dimensions.
        UserWarning
            If TDI data has been set previously, the instance will be reset.

        Notes
        -----
        The length of input ``tdi_data_array`` must match self.data_info.frequency_series_length.
        If the frequency series are obtained from FFT, self.data_info.frequency_mask_array
        may be needed to mask the array before input.
        """
        if self._reset_flag:
            warnings.warn(
                "You are setting `fd_data` with input array, whereas you have probably "
                "set TDI data of current instance previously. Please check whether this "
                "is intertional. \n"
                "In order to avoid potential errors, current instance is reset. Please "
                "regenerate TDI data of other domian or noise behavior data if needed. "
            )
            self._reset()

        self.set_data_info(
            channels,
            duration,
            delta_time,
            start_time,
            minimum_frequency,
            maximum_frequency,
        )
        channels_num, samples_num = tdi_data_array.shape
        if not len(channels) == channels_num:
            raise ValueError(
                f"You set channenls with {channels}, while the length of first dimension "
                f"of input array is {channels_num}."
            )
        if not self.data_info.frequency_series_length == samples_num:
            raise ValueError(
                f"The length of second dimension of input array is {samples_num}. It is "
                f"different with the `frequency_series_length={self.data_info.frequency_series_length}` "
                f"which is set according to the duration, delta_time, minimum and maximum frequency. \n"
                f"You may need to crop the tdi_data_array with `TDIChannelData.data_info.frequency_mask_array` "
                f"before input. Please check the input again or open an issue."
            )

        self._init_fd_data()
        self.fd_data.from_numpy(
            dict(
                zip(
                    channels,
                    np.stack(
                        (
                            np.astype(tdi_data_array.real, self._np_fp),
                            np.astype(tdi_data_array.imag, self._np_fp),
                        ),
                        axis=-1,
                    ),
                )
            )
        )

        self._reset_flag = True

    def set_td_data_from_zero(
        self,
        channels: tuple[str, ...],
        duration: float,
        delta_time: float,
        start_time: float = 0.0,
        minimum_frequency: float = 1e-5,
        maximum_frequency: float = 0.1,
    ) -> None:
        """
        Initialize time domain TDI data with zeros.

        Parameters
        ----------
        channels : tuple[str, ...]
            TDI channel names.
        duration : float
            Observing duration of data, in seconds.
        delta_time : float
            Sampling cadence, in seconds.
        start_time : float, optional
            Time label for the first time sample, in seconds. Default is 0.0.
        minimum_frequency : float, optional
            Minimum frequency for the limited frequency band, in Hz. Default is 1e-5.
        maximum_frequency : float, optional
            Maximum frequency for the limited frequency band, in Hz. Default is 0.1.

        Raises
        ------
        UserWarning
            If TDI data has been set previously, the instance will be reset.
        """
        if self._reset_flag:
            warnings.warn(
                "You are setting `td_data` with zero value, whereas you have probably "
                "set TDI data of current instance previously. Please check whether this "
                "is intertional. \n"
                "In order to avoid potential errors, current instance is reset. Please "
                "regenerate TDI data of other domian or noise behavior data if needed. "
            )
            self._reset()

        self.set_data_info(
            channels,
            duration,
            delta_time,
            start_time,
            minimum_frequency,
            maximum_frequency,
        )

        self._init_td_data()
        self.td_data.fill(0.0)

        self._reset_flag = True

    def set_fd_data_from_zero(
        self,
        channels: tuple[str, ...],
        duration: float,
        delta_time: float,
        start_time: float = 0.0,
        minimum_frequency: float = 1e-5,
        maximum_frequency: float = 0.1,
    ) -> None:
        """
        Initialize frequency domain TDI data with zeros.

        Parameters
        ----------
        channels : tuple[str, ...]
            TDI channel names.
        duration : float
            Observing duration of data, in seconds.
        delta_time : float
            Sampling cadence, in seconds.
        start_time : float, optional
            Time label for the first time sample, in seconds. Default is 0.0.
        minimum_frequency : float, optional
            Minimum frequency for the limited frequency band, in Hz. Default is 1e-5.
        maximum_frequency : float, optional
            Maximum frequency for the limited frequency band, in Hz. Default is 0.1.

        Raises
        ------
        UserWarning
            If TDI data has been set previously, the instance will be reset.
        """
        if self._reset_flag:
            warnings.warn(
                "You are setting `fd_data` with zero value, whereas you have probably "
                "set TDI data of current instance previously. Please check whether this "
                "is intertional. \n"
                "In order to avoid potential errors, current instance is reset. Please "
                "regenerate TDI data of other domian or noise behavior data if needed. "
            )
            self._reset()

        self.set_data_info(
            channels,
            duration,
            delta_time,
            start_time,
            minimum_frequency,
            maximum_frequency,
        )

        self._init_fd_data()
        self.fd_data.fill(0.0)

        self._reset_flag = True

    def set_wd_data_from_zero(self) -> None:
        """
        Initialize wavelet domain TDI data with zeros.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def set_fd_data_from_td(
        self,
        window: None | float | str | tuple[str | float] | NDArray = None,
    ) -> None:
        """
        Compute frequency domain data from time domain data using FFT.

        Parameters
        ----------
        window : None, float, str, tuple, or NDArray, optional
            Window function to apply before FFT. See ``scipy.signal.get_window`` for details.
            If None, no windowing is applied. Default is None.

        Raises
        ------
        ValueError
            If td_data is not set or fd_data has already been set.

        Notes
        -----

        - time-shift of start time is not applied in current implementation.
        - The FFT is normalized by dividing by the sampling frequency.
        """

        if (self.td_data is None) or (self.fd_data is not None):
            raise ValueError(
                "Fourier transform cannot be excuted since the `td_data` is not set or "
                "`fd_data` has been set previously."
            )
        else:
            self._init_fd_data()

            if window is None:
                weight = np.ones(self.data_info.time_series_length)
            elif isinstance(window, np.ndarray):
                weight = window
            else:
                weight = signal.get_window(window, self.data_info.time_series_length)

            fd_data_numpy = dict.fromkeys(self.data_info.channels)
            td_data_numpy = self.td_data.to_numpy()

            # start_time_shift = np.exp(
            #     -1j
            #     * 2
            #     * PI
            #     * self.data_info.time_samples_array[0]
            #     * self.data_info.full_frequency_samples_array
            # )

            for chan in self.data_info.channels:
                td_data_chan = td_data_numpy[chan]
                windowed_data = td_data_chan * weight
                fd_data_chan = np.fft.rfft(windowed_data)
                # fd_data_chan *= start_time_shift / self.data_info.sampling_frequency
                fd_data_chan /= self.data_info.sampling_frequency
                fd_data_chan = fd_data_chan[self.data_info.frequency_mask_array]
                fd_data_numpy[chan] = np.astype(fd_data_chan, self._np_fp)

            complex_numpy_array_dict_to_taichi_field(fd_data_numpy, self.fd_data)

    def set_td_data_from_fd(
        self,
        window: None | float | str | tuple[str | float] | NDArray = None,
    ) -> None:
        """
        Compute time domain data from frequency domain data using inverse FFT.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def set_wd_data_from_td(self) -> None:
        """
        Compute wavelet domain data from time domain data.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def set_wd_data_from_fd(self) -> None:
        """
        Compute wavelet domain data from frequency domain data.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def set_fd_noise_power_density_from_td_data(self) -> None:
        """
        Estimate frequency domain noise power spectral density from time domain data.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def set_fd_noise_power_density_from_model(
        self,
        noise_model: str | FrequencyDomainNoiseModel,
        **model_kwards: dict,
    ) -> None:
        """
        Set frequency domain noise power spectral density from a noise model.

        Parameters
        ----------
        noise_model : str or FrequencyDomainNoiseModel
            Name of an available noise model or a FrequencyDomainNoiseModel instance.
        **model_kwards : dict
            Additional keyword arguments passed to the noise model.

        Raises
        ------
        ValueError
            If ``data_info`` has not been set, or if the ``noise_model`` string is not recognized.
        UserWarning
            If ``fd_noise_power_density`` has been set previously.
        """
        if isinstance(noise_model, str):
            if noise_model in available_noise_models.keys():
                psd = available_noise_models[noise_model]
            else:
                raise ValueError(
                    f"{noise_model} is not a implemented noise model. \n"
                    f"Current available noise models including {[*available_noise_models.keys()]}."
                )
        elif isinstance(noise_model, FrequencyDomainNoiseModel):
            psd = noise_model

        if self.data_info is None:
            raise ValueError(
                "The `data_info` has not yet set. Can not obtain `frequency_series_length` "
                "to initialize `fd_noise_power_density` Please first set TDI data in any "
                "domain or call directly `set_data_info`."
            )
        if self.fd_noise_power_density is not None:
            warnings.warn(
                "You are setting `fd_noise_power_density` for current instance, whereas "
                "it have been set previously. It will be reset and updated, please make "
                "sure the updated noise power density is consistent with the stored TDI data."
            )
        self.fd_noise_power_density = ti.Struct.field(
            dict.fromkeys(self.data_info.channels, float),
            shape=(self.data_info.frequency_series_length,),
        )
        self.fd_noise_power_density.from_numpy(
            psd(
                self.data_info.frequency_samples_array,
                self.data_info.channels,
                self.scaling,
                **model_kwards,
            )
        )

    def get_td_noise_realization(self):
        """
        Generate a noise realization in time domain.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def get_fd_noise_realization(
        self, seed=None, output_type: str = "taichi"
    ) -> ti.StructField | dict[str, NDArray]:
        """
        Generate a noise realization in frequency domain from the power spectral density.

        This method generates colored Gaussian noise in the frequency domain based on
        the stored noise power spectral density. The noise is generated as white noise
        and then colored by multiplying with the square root of the PSD.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator for reproducible results. Default is None.
        output_type : str, optional
            Output format, either "taichi" for ``taichi.StructField`` or "numpy" for dict of arrays.
            Default is "taichi".

        Returns
        -------
        ti.StructField or dict[str, NDArray]
            Noise realization in frequency domain. Format depends on ``output_type`` parameter.

        Raises
        ------
        ValueError
            If ``fd_noise_power_density`` has not been set, or if ``output_type`` is invalid.

        References
        ----------
        .. [1] Eq. 12 in Phys. Rev. D 102, 023033 (2020)
               https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.023033
        .. [2] Bilby PowerSpectralDensity.get_noise_realisation
               https://lscsoft.docs.ligo.org/bilby/api/bilby.gw.detector.psd.PowerSpectralDensity.html#bilby.gw.detector.psd.PowerSpectralDensity.get_noise_realisation

        Notes
        -----
        To avoid directly modifying the stored tdi_data internally, which could potentially
        lead to mismatch among data of different domains, this method only returns the
        generated noise data. Use ``add_into_fd_data`` to manually add the noise realization
        into the TDI data.
        """
        if self.fd_noise_power_density is None:
            raise ValueError(
                "Setting `fd_noise_power_density` before generating noise realization."
            )

        rng = np.random.default_rng(seed=seed)
        var = 0.5 / (self.data_info.delta_frequency) ** 0.5
        noise = {}

        for chan in self.data_info.channels:
            # generate white noise
            re, im = rng.normal(
                0, var, (2, self.data_info.full_frequency_series_length)
            )
            # set DC component
            re[0] = 0.0
            im[0] = 0.0
            # set Nyquist frequency component for ensuring the Hermitian symmetry
            if np.mod(self.data_info.time_series_length, 2) == 0:
                im[-1] = 0.0
                re[-1] = 0.0
            noise_chan = (re + 1j * im)[
                self.data_info.frequency_mask_array
            ] * self.fd_noise_power_density_numpy[chan] ** 0.5
            noise[chan] = np.astype(noise_chan, self._np_cp)

        if output_type == "taichi":
            ret = ti.Struct.field(
                dict.fromkeys(self.data_info.channels, ti_complex),
                shape=(self.data_info.frequency_series_length,),
            )
            complex_numpy_array_dict_to_taichi_field(noise, ret)
        elif output_type == "numpy":
            ret = noise
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        return ret

    def add_into_td_data(self) -> None:
        """
        Add data into time domain TDI data.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def add_into_fd_data(self, input: ti.StructField | dict[str, NDArray]) -> None:
        """
        Add input data into frequency domain TDI data.

        Parameters
        ----------
        input : ti.StructField or dict[str, NDArray]
            Input data to add. Can be either a ``taichi.StructField`` or a dictionary
            of numpy arrays. Must have the same shape and channels as ``fd_data``.

        Raises
        ------
        ValueError
            If input shape or channels do not match ``fd_data``.
        TypeError
            If input type is not ``taichi.StructField`` or ``dict[str, NDArray]``.
        """
        if isinstance(input, ti.StructField):
            if not input.shape == (self.data_info.frequency_series_length,):
                raise ValueError(
                    "Cannot add the input StructField into the `fd_data`, since the shape "
                    "of input is different with the `fd_data`"
                )
            if not set(input.keys) == set(self.data_info.channels):
                raise ValueError(
                    "Cannot add the input StructField into the `fd_data`, since the channnels "
                    "contained by input is different with the `fd_data`"
                )
            input_field = input

        elif isinstance(input, dict):
            if not all(
                [
                    len(data) == self.data_info.frequency_series_length
                    for _, data in input.items()
                ]
            ):
                raise ValueError(
                    "Cannot add the input dict of array into the `fd_data`, since there "
                    "is at least one array in the input dict having different length with "
                    "the TDI data."
                )
            if not set(input.keys()) == set(self.data_info.channels):
                raise ValueError(
                    "Cannot add the input dict of array into the `fd_data`, since the channnels "
                    "contained by input is different with the TDI data"
                )
            input_field = ti.Struct.field(
                dict.fromkeys(self.data_info.channels, ti_complex),
                shape=(self.data_info.frequency_series_length,),
            )
            complex_numpy_array_dict_to_taichi_field(input, input_field)

        else:
            raise TypeError("Unsupported type, expect ti.StructField or dict[NDArray]")

        _add_input_field_into_tdi_data(self.fd_data, input_field)

    def add_into_wd_data(self) -> None:
        """
        Add data into wavelet domain TDI data.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError()

    def save_to_file(self, filename: str) -> None:
        """
        Save the data stored in the instance to an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the output HDF5 file.

        Notes
        -----
        The file will contain three groups:
        
        - data_info: metadata about the TDI data
        - tdi_data: TDI data in various domains (``td_data``, ``fd_data``, ``wd_data``)
        - noise_data: noise power spectral densities
        """
        tdi_data = [
            "td_data",
            "fd_data",
            "wd_data",
        ]
        noise_data = [
            "fd_noise_power_density",
            "wd_noise_power_density",
        ]

        with h5py.File(filename, "x") as file:
            for key, data in asdict(self.data_info).items():
                if key == "channels":
                    file.create_dataset(
                        f"data_info/{key}",
                        data=np.array(data, dtype=h5py.string_dtype(encoding="utf-8")),
                    )
                else:
                    file.create_dataset(f"data_info/{key}", data=np.array(data))

            for key in tdi_data:
                data = getattr(self, key)
                if data is not None:
                    for chan, chan_array in data.to_numpy().items():
                        file.create_dataset(f"tdi_data/{key}/{chan}", data=chan_array)

            for key in noise_data:
                data = getattr(self, key)
                if data is not None:
                    for chan, chan_array in data.to_numpy().items():
                        file.create_dataset(f"noise_data/{key}/{chan}", data=chan_array)

    @staticmethod
    def recover_from_file(filename: str) -> "TDIChannelData":
        """
        Recover TDIChannelData instance from an HDF5 file.

        This method reconstructs a ``TDIChannelData`` instance from a file saved by
        the ``save_to_file`` method.

        Parameters
        ----------
        filename : str
            Path to the input HDF5 file.

        Returns
        -------
        TDIChannelData
            Recovered ``TDIChannelData`` instance with all stored data.

        Raises
        ------
        AssertionError
            If any derived quantities in the file do not match the recomputed values.

        Notes
        -----
        For other data formats, use methods for setting from input arrays according
        to the domain of the data, such as ``set_td_data_from_input``, etc.
        """
        ret_cls = TDIChannelData()

        with h5py.File(filename, "r") as file:
            channels = tuple(file["data_info/channels"][()].astype(str))
            duration = float(file["data_info/duration"][()])
            delta_time = float(file["data_info/delta_time"][()])
            start_time = float(file["data_info/start_time"][()])
            minimum_frequency = float(file["data_info/minimum_frequency"][()])
            maximum_frequency = float(file["data_info/maximum_frequency"][()])

            ret_cls.set_data_info(
                channels,
                duration,
                delta_time,
                start_time,
                minimum_frequency,
                maximum_frequency,
            )

            assert ret_cls.data_info.sampling_frequency == float(
                file["data_info/sampling_frequency"][()]
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `sampling_frequency` is different from the stored value."
            )
            assert ret_cls.data_info.delta_frequency == float(
                file["data_info/delta_frequency"][()]
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `delta_frequency` is different from the stored value."
            )
            assert ret_cls.data_info.time_series_length == int(
                file["data_info/time_series_length"][()]
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `time_series_length` is different from the stored value."
            )
            assert np.array_equal(
                ret_cls.data_info.time_samples_array,
                file["data_info/time_samples_array"][()],
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `time_samples_array` is different from the stored value."
            )
            assert ret_cls.data_info.full_frequency_series_length == int(
                file["data_info/full_frequency_series_length"][()]
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `full_frequency_series_length` is different from the stored value."
            )
            assert np.array_equal(
                ret_cls.data_info.full_frequency_samples_array,
                file["data_info/full_frequency_samples_array"][()],
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `full_frequency_samples_array` is different from the stored value."
            )
            assert np.array_equal(
                ret_cls.data_info.frequency_mask_array,
                file["data_info/frequency_mask_array"][()],
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `frequency_mask_array` is different from the stored value."
            )
            assert np.array_equal(
                ret_cls.data_info.frequency_samples_array,
                file["data_info/frequency_samples_array"][()],
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `frequency_samples_array` is different from the stored value."
            )
            assert ret_cls.data_info.frequency_series_length == int(
                file["data_info/frequency_series_length"][()]
            ), (
                f"Failed to recover TDIChannelData from file {filename}, "
                "the `frequency_series_length` is different from the stored value."
            )

            tdi_data_group = file["tdi_data"]
            if "td_data" in tdi_data_group:
                ret_cls._init_td_data()
                ret_cls.td_data.from_numpy(
                    {
                        chan: dataset[()]
                        for chan, dataset in tdi_data_group["td_data"].items()
                    }
                )
            if "fd_data" in tdi_data_group:
                ret_cls._init_fd_data()
                ret_cls.fd_data.from_numpy(
                    {
                        chan: dataset[()]
                        for chan, dataset in tdi_data_group["fd_data"].items()
                    }
                )
            if "wd_data" in tdi_data_group:
                ret_cls._init_wd_data()
                ret_cls.wd_data.from_numpy(
                    {
                        chan: dataset[()]
                        for chan, dataset in tdi_data_group["wd_data"].items()
                    }
                )

            noise_data_group = file["noise_data"]
            if "fd_noise_power_density" in noise_data_group:
                ret_cls.fd_noise_power_density = ti.Struct.field(
                    dict.fromkeys(ret_cls.data_info.channels, float),
                    shape=(ret_cls.data_info.frequency_series_length,),
                )
                ret_cls.fd_noise_power_density.from_numpy(
                    {
                        chan: dataset[()]
                        for chan, dataset in noise_data_group[
                            "fd_noise_power_density"
                        ].items()
                    }
                )
            if "wd_noise_power_density" in noise_data_group:
                raise NotImplementedError()

        ret_cls._reset_flag = True  # remember to mark the update state

        return ret_cls

    @property
    def td_data_numpy(self) -> dict[str, NDArray]:
        """
        Get time domain data as numpy arrays.

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping channel names to time domain data arrays.
        """
        return self.td_data.to_numpy()

    @property
    def fd_data_numpy(self) -> dict[str, NDArray]:
        """
        Get frequency domain data as complex numpy arrays.

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping channel names to complex frequency domain data arrays.
        """
        return taichi_field_to_complex_numpy_array_dict(self.fd_data)

    @property
    def wd_data_numpy(self) -> dict[str, NDArray]:
        """
        Get wavelet domain data as numpy arrays.

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping channel names to wavelet domain data arrays.
        """
        return self.wd_data.to_numpy()

    @property
    def fd_noise_power_density_numpy(self) -> dict[str, NDArray]:
        """
        Get frequency domain noise power spectral density as numpy arrays.

        Returns
        -------
        dict[str, NDArray]
            Dictionary mapping channel names to noise PSD arrays.
        """
        return self.fd_noise_power_density.to_numpy()


class TDICombinationModel(ABC):
    """
    Abstract base class for TDI combination models.

    This class defines the interface for TDI combination models that compute
    TDI observables from single-link responses.
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """
        Get the domain of the TDI combination.

        Returns
        -------
        str
            Domain identifier, either 'td' (time domain) or 'fd' (frequency domain).
        """
        pass

    @abstractmethod
    def update_tdi_response(self) -> None:
        """
        Update TDI response from single-link responses.

        This method computes TDI observables from the detector's single-link responses.
        """
        pass


@ti.data_oriented
class TDMichelsonConstantEqualArm(TDICombinationModel):
    """
    Time-domain Michelson TDI combination for constant equal-arm detectors (**UNCOMPLETE**).

    This class implements TDI combinations in the time domain for space-borne
    interferometers with constant and equal arm lengths. Supports both
    generation 1.5 and 2.0 TDI, and both orthogonal (A, E, T) and
    non-orthogonal (X, Y, Z) channel combinations.
    """

    domain = "td"

    def __init__(self, generation="1.5", orthogonal=True):
        """
        Initialize time-domain Michelson TDI combination model.

        Parameters
        ----------
        generation : str, optional
            TDI generation, either '1.5' or '2.0'. Default is '1.5'.
        orthogonal : bool, optional
            Whether to use orthogonal channels (A, E, T) or non-orthogonal (X, Y, Z).
            Default is True.

        Raises
        ------
        ValueError
            If generation is not '1.5' or '2.0'.
        """
        self.generation = str(generation)
        if self.generation == "1.5":  # TODO: improve for unconstant armlength
            self.max_num_delay = 3
        elif self.generation == "2.0":
            self.max_num_delay = 7
        else:
            raise ValueError(f"Unsupported generation {self.generation}.")

        self.orthogonal = bool(orthogonal)
        if self.orthogonal:
            self.labels = ("A", "E", "T")
        else:
            self.labels = ("X", "Y", "Z")

        self.added_time_samples_number = None
        self.extended_time_series_length = None

    def init_tdi_combination_model(self, detector: "InterferometerAntenna") -> None:
        """
        Initialize the TDI combination model with a detector instance.

        Parameters
        ----------
        detector : InterferometerAntenna
            Detector instance to associate with this TDI model.
        """
        self.detector = weakref.proxy(detector)
        self.detector.tdi_response = ti.Struct.field(
            dict.fromkeys(self.labels, float),
            shape=(self.detector.tdi_data.data_info.time_series_length,),
        )
        self.added_time_samples_number = int(
            self.max_num_delay
            * self.detector.orbit_model.armlength_sec
            // self.detector.tdi_data.data_info.delta_time
            + 1
        )
        self.extended_time_series_length = int(
            self.detector.tdi_data.data_info.time_series_length
            + self.added_time_samples_number
        )

    @ti.kernel
    def update_tdi_response(self):
        """Update TDI response from single-link responses in time domain."""
        # temporarily store the single link response with delays in the loop
        delayed_response = ti.field(SingleLinkStructReal, shape=(self.max_num_delay,))

        # displeasement of time samples after delays, in the range of [0, 1].
        # constant for each delay in the case of equal-arm, can be computed and cached before the loop.
        # note the num_delay = idx + 1
        t_frac = ti.field(float, shape=(self.max_num_delay,))
        for i in ti.static(range(self.max_num_delay)):
            num_delay = i + 1
            t_frac[i] = (
                1.0
                - (
                    num_delay
                    * self.detector.orbit_model.armlength_sec
                    % self.detector.tdi_data.data_info.delta_time
                )
                / self.detector.tdi_data.data_info.delta_time
            )

        for i in self.detector.tdi_response:
            # compute delayed single-link responses
            for num_delay in ti.static(range(self.max_num_delay, 0, -1)):  # using reverse order here for more continuously accessing single_link_response field. # fmt: skip
                i_shift = (
                    (self.max_num_delay - num_delay)
                    * self.detector.orbit_model.armlength_sec
                    // self.detector.tdi_data.data_info.delta_time
                )
                links_left = self.detector.single_link_response[i + i_shift]
                links_right = self.detector.single_link_response[i + i_shift + 1]

                for label in ti.static(
                    ["link12", "link21", "link23", "link32", "link31", "link13"]
                ):
                    delayed_response[num_delay - 1][label] = linear_interpolate_kernel(
                        links_left[label],
                        links_right[label],
                        t_frac[num_delay - 1],
                    )

            links_response = self.detector.single_link_response[i + self.added_time_samples_number]  # fmt: skip
            X = 0.0
            Y = 0.0
            Z = 0.0
            if ti.static(self.generation == "1.5"):
                X = (
                    links_response.link13
                    + delayed_response.link31[0]
                    + delayed_response.link12[1]
                    + delayed_response.link21[2]
                    - links_response.link12
                    - delayed_response.link21[0]
                    - delayed_response.link13[1]
                    - delayed_response.link31[2]
                )
                Y = (
                    links_response.link21
                    + delayed_response.link12[0]
                    + delayed_response.link23[1]
                    + delayed_response.link32[2]
                    - links_response.link23
                    - delayed_response.link32[0]
                    - delayed_response.link21[1]
                    - delayed_response.link12[2]
                )
                Z = (
                    links_response.link32
                    + delayed_response.link23[0]
                    + delayed_response.link31[1]
                    + delayed_response.link13[2]
                    - links_response.link31
                    - delayed_response.link13[0]
                    - delayed_response.link32[1]
                    - delayed_response.link23[2]
                )
            elif ti.static(self.generation == "2.0"):
                X = (
                    links_response.link13
                    + delayed_response.link31[0]
                    + delayed_response.link12[1]
                    + delayed_response.link21[2]
                    + delayed_response.link12[3]
                    + delayed_response.link21[4]
                    + delayed_response.link13[5]
                    + delayed_response.link31[6]
                    - links_response.link12
                    - delayed_response.link21[0]
                    - delayed_response.link13[1]
                    - delayed_response.link31[2]
                    - delayed_response.link13[3]
                    - delayed_response.link31[4]
                    - delayed_response.link12[5]
                    - delayed_response.link21[6]
                )
                Y = (
                    links_response.link21
                    + delayed_response.link12[0]
                    + delayed_response.link23[1]
                    + delayed_response.link32[2]
                    + delayed_response.link23[3]
                    + delayed_response.link32[4]
                    + delayed_response.link21[5]
                    + delayed_response.link12[6]
                    - links_response.link23
                    - delayed_response.link32[0]
                    - delayed_response.link21[1]
                    - delayed_response.link12[2]
                    - delayed_response.link21[3]
                    - delayed_response.link12[4]
                    - delayed_response.link23[5]
                    - delayed_response.link32[6]
                )
                Z = (
                    links_response.link32
                    + delayed_response.link23[0]
                    + delayed_response.link31[1]
                    + delayed_response.link13[2]
                    + delayed_response.link31[3]
                    + delayed_response.link13[4]
                    + delayed_response.link32[5]
                    + delayed_response.link23[6]
                    - links_response.link31
                    - delayed_response.link13[0]
                    - delayed_response.link32[1]
                    - delayed_response.link23[2]
                    - delayed_response.link32[3]
                    - delayed_response.link23[4]
                    - delayed_response.link31[5]
                    - delayed_response.link13[6]
                )

            if ti.static(self.orthogonal):
                self.detector.tdi_response[i].A = (Z - X) / tm.sqrt(2.0)
                self.detector.tdi_response[i].E = (X - 2.0 * Y + Z) / tm.sqrt(6.0)
                self.detector.tdi_response[i].T = (X + Y + Z) / tm.sqrt(3.0)
            else:
                self.detector.tdi_response[i].X = X
                self.detector.tdi_response[i].Y = Y
                self.detector.tdi_response[i].Z = Z


@ti.data_oriented
class FDMichelsonConstantEqualArm(TDICombinationModel):
    """
    Frequency-domain Michelson TDI combination for constant equal-arm detectors.

    This class implements TDI combinations in the frequency domain for space-based
    interferometers with constant and equal arm lengths. Supports both
    generation 1.5 and 2.0 TDI, and both orthogonal (A, E, T) and
    non-orthogonal (X, Y, Z) channel combinations.
    """

    domain = "fd"

    def __init__(self, generation="1.5", orthogonal=True):
        """
        Initialize frequency-domain Michelson TDI combination model.

        Parameters
        ----------
        generation : str, optional
            TDI generation, either '1.5' or '2.0'. Default is '1.5'.
        orthogonal : bool, optional
            Whether to use orthogonal channels (A, E, T) or non-orthogonal (X, Y, Z).
            Default is True.

        Raises
        ------
        ValueError
            If generation is not '1.5' or '2.0'.
        """
        self.generation = str(generation)
        if not (self.generation == "1.5" or self.generation == "2.0"):
            raise ValueError(f"Unsupported generation {self.generation}.")

        self.orthogonal = bool(orthogonal)
        if self.orthogonal:
            self.labels = ("A", "E", "T")
        else:
            self.labels = ("X", "Y", "Z")

        self.detector = None
        self._cached_field = None

    def init_tdi_combination_model(self, detector: "InterferometerAntenna") -> None:
        """
        Initialize the TDI combination model with a detector instance.

        Parameters
        ----------
        detector : InterferometerAntenna
            Detector instance to associate with this TDI model.
        """
        self.detector = weakref.proxy(detector)
        self.detector.tdi_response = ti.Struct.field(
            dict.fromkeys(self.labels, ti_complex),
            shape=(self.detector.tdi_data.data_info.frequency_series_length,),
        )
        self._cached_field = ti.Struct.field(
            {"prefactor": ti_complex, "delay_factor": ti_complex},
            shape=(self.detector.tdi_data.data_info.frequency_series_length,),
        )
        self._set_cached_field()

    @ti.kernel
    def _set_cached_field(self):
        """
        Compute and cache frequency-dependent prefactors and delay factors.

        This method computes and caches the delay factors :math:`\\mathrm{exp}(-i2\\pi fL)`
        and generation-dependent prefactors for TDI computation.
        """
        for i in self._cached_field:
            delay_factor = tm.cexp(
                -2.0
                * PI
                * self.detector.tdi_data.frequency_samples[i]
                * self.detector.orbit_model.arm_length_sec
                * ti_complex([0.0, 1.0])
            )

            prefactor = ti_complex([0.0, 0.0])
            if ti.static(self.generation == "1.5"):
                prefactor = ti_complex([1.0, 0.0]) - tm.cpow(delay_factor, 2)
            elif ti.static(self.generation == "2.0"):
                prefactor = (
                    ti_complex([1.0, 0.0])
                    - tm.cpow(delay_factor, 2)
                    - tm.cpow(delay_factor, 4)
                    + tm.cpow(delay_factor, 6)
                )
            self._cached_field[i]["prefactor"] = prefactor
            self._cached_field[i]["delay_factor"] = delay_factor

    @staticmethod
    @ti.func
    def _get_X_channel_response(
        delay_factor: ti.template(), singlelink_response: ti.template()
    ) -> ti_complex:
        """
        Compute X channel of TDI combination in frequency domain.

        Parameters
        ----------
        delay_factor : ti.template()
            Delay factor, :math:`\\mathrm{exp}(-i2\\pi fL)`.
        singlelink_response : ti.template()
            Single-link responses for six links.

        Returns
        -------
        ti_complex
            X channel response without the generation prefactor.
        """
        return (
            singlelink_response["link13"]
            - singlelink_response["link12"]
            + tm.cmul(
                delay_factor,
                (singlelink_response["link31"] - singlelink_response["link21"]),
            )
        )

    @staticmethod
    @ti.func
    def _get_Y_channel_response(
        delay_factor: ti.template(), singlelink_response: ti.template()
    ) -> ti_complex:
        """
        Compute Y channel of TDI combination in frequency domain.

        Parameters
        ----------
        delay_factor : ti.template()
            Delay factor, :math:`\\mathrm{exp}(-i2\\pi fL)`.
        singlelink_response : ti.template()
            Single-link responses for all six links.

        Returns
        -------
        ti_complex
            Y channel response without the generation prefactor.
        """
        return (
            singlelink_response["link21"]
            - singlelink_response["link23"]
            + tm.cmul(
                delay_factor,
                (singlelink_response["link12"] - singlelink_response["link32"]),
            )
        )

    @staticmethod
    @ti.func
    def _get_Z_channel_response(
        delay_factor: ti.template(), singlelink_response: ti.template()
    ) -> ti_complex:
        """
        Compute Z channel of TDI combination in frequency domain.

        Parameters
        ----------
        delay_factor : ti.template()
            Delay factor, :math:`\\mathrm{exp}(-i2\\pi fL)`.
        singlelink_response : ti.template()
            Single-link responses for all six links.

        Returns
        -------
        ti_complex
            Z channel response without the generation prefactor.
        """
        return (
            singlelink_response["link32"]
            - singlelink_response["link31"]
            + tm.cmul(
                delay_factor,
                (singlelink_response["link23"] - singlelink_response["link13"]),
            )
        )

    @ti.kernel
    def update_tdi_response(self):
        """Update TDI response from single-link responses in frequency domain."""
        for i in self.detector.tdi_response:
            prefactor = self._cached_field[i].prefactor
            delay_factor = self._cached_field[i].delay_factor
            singlelink_response = self.detector.single_link_response[i]
            X = tm.cmul(
                prefactor,
                self._get_X_channel_response(delay_factor, singlelink_response),
            )
            Y = tm.cmul(
                prefactor,
                self._get_Y_channel_response(delay_factor, singlelink_response),
            )
            Z = tm.cmul(
                prefactor,
                self._get_Z_channel_response(delay_factor, singlelink_response),
            )
            if ti.static(self.orthogonal):
                A = (Z - X) / tm.sqrt(2.0)
                E = (X - 2.0 * Y + Z) / tm.sqrt(6.0)
                T = (X + Y + Z) / tm.sqrt(3.0)
                self.detector.tdi_response[i].A = A
                self.detector.tdi_response[i].E = E
                self.detector.tdi_response[i].T = T
            else:
                self.detector.tdi_response[i].X = X
                self.detector.tdi_response[i].Y = Y
                self.detector.tdi_response[i].Z = Z


@ti.data_oriented
class FDMichelsonConstantEqualArmFFT(TDICombinationModel):
    """
    Frequency-domain Michelson TDI combination using FFT approach.

    This class is a placeholder for implementing TDI combinations in the frequency
    domain using FFT-based methods for constant equal-arm detectors.

    Notes
    -----
    This class is not yet implemented.
    """
    pass


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .antenna import InterferometerAntenna
