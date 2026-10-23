"""Likelihood computation with jax."""

from collections import namedtuple
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..waveform.interface_ripple import WaveformRippleBase


class DetectorData(NamedTuple):
    """Static per-detector data for the likelihood."""

    data: jnp.ndarray
    psd: jnp.ndarray
    df: float
    channel_indices: tuple | None


def noise_weighted_inner_product(observed_data, response_data, psd, df):
    """
    The noise-weighted inner product under the Gaussian stationary noise
    assumption (Whittle likelihood kernel ``-2 * df * sum(|data-model|^2 / psd)``).

    Parameters
    ----------
    observed_data, response_data, psd : jnp.ndarray
        Observed frequency-domain TDI channel data, modeled response, and noise
        power spectral density, each of shape ``(N_freq,)`` or
        ``(N_channels, N_freq)``.
    df : float
        Frequency bin spacing in Hz.

    Returns
    -------
    float
        Value for the Whittle log-likelihood, summed over all axes.
    """
    residual = observed_data - response_data
    log_l = jnp.sum(jnp.real(residual * residual.conj()) / psd)
    return -2.0 * df * log_l


class WhittleLikelihoodJax:
    """The Whittle likelihood computation with jax."""

    def __init__(self, waveform, detector, channels) -> None:
        """
        Parameters
        ----------
        waveform : WaveformRippleBase
            Waveform interface instance for waveform generation, the function of waveform
            computation should be jit-able and grad-able.
            Its ``frequencies`` should match the ``frequency_samples`` of the TDI data
            of all detectors.
        detector : InterferometerAntenna or tuple of InterferometerAntenna
            Detector instance(s) for computing detector responses, should be
            configured with ``backend='jax'``, with frequency-domain TDI data
            and noise power spectral densities properly set.
        channels : tuple of str
            TDI channels used in likelihood computation. Must be a subset of
            ('A', 'E', 'T'). Common choices are ('A', 'E', 'T') or ('A', 'E').

        Raises
        ------
        TypeError
            If ``waveform`` is not a :class:`WaveformRippleBase` instance.
        ValueError
            - If any detector does not use ``backend='jax'``;
            - If any channel in ``channels`` is not one of orthogonal 'A', 'E', or 'T';
            - If any requested channel is not provided by the TDI combination model;
            - If the frequency-domain data or noise power spectral density of a detector
              is not set or does not contain all requested channels.

        Notes
        -----
        - The waveform computation function returns a dict: ``{"plus": hp, "cross": hc,
          "tf": tf}`` for a single-mode waveform, or a mode-keyed dict for higher modes.
        - The coalescence time shift is applied in the response computation.
          Thus, the coalescence time should be set to zero in the waveform
          computation.
        """
        if not isinstance(waveform, WaveformRippleBase):
            raise TypeError(
                "WhittleLikelihoodJax requires the waveform to be a "
                "WaveformRippleBase instance, but got "
                f"{type(waveform).__name__}."
            )

        if not isinstance(detector, tuple):
            detector = (detector,)

        for det in detector:
            if not (det.backend == "jax"):
                raise ValueError(
                    "WhittleLikelihoodJax requires all detectors configured with backend='jax', "
                    f"but got backend={det.backend} for detector={det.name}."
                )
            if not jnp.array_equal(
                det.tdi_data.frequency_samples, waveform.frequencies
            ):
                raise ValueError(
                    "The waveform frequencies should match the frequency samples "
                    f"of the TDI data for detector={det.name}."
                )

        unsupported_chans = [chan for chan in channels if chan not in ("A", "E", "T")]
        if any(unsupported_chans):
            raise ValueError(
                "The likelihood computation only supports orthogonal channels ('A', 'E', 'T'). "
                f"{unsupported_chans} are not supported currently."
            )

        self.waveform = waveform
        self.detector = detector
        self.channels = tuple(channels)

        response_params = (
            "ecliptic_longitude",
            "ecliptic_latitude",
            "polarization",
            "coalescence_time",
        )
        self.likelihood_param_type = namedtuple(
            "LikelihoodParams", (*waveform.required_params, *response_params)
        )

        self._static_data: tuple[DetectorData, ...] = ()
        self._prepare_static_data()

    def _prepare_static_data(self):
        """
        Prepare the static data of detectors for likelihood computation, facilitating
        capturing these data via closure in jit-compile of :meth:`log_likelihood`.

        Observed frequency-domain TDI data and PSD data are stacked with the shape
        ``(N_channels, N_freq)`` and ordered as the requested channels. Along with the
        ``delta_frequency``, and the indices for selecting the requested channels from
        the TDI combination output (``None`` when no slicing is needed), are stored in
        a namedtuple for each detector.

        The namedtuples for all detectors are collected in a tuple and set as the
        :attr:`_static_data` attribute.

        Raises
        ------
        ValueError
            - If any requested channel is not a label provided in the TDI combination
            model;
            - If the frequency-domain data or noise power spectral density is not set or
              does not contain all requested channels.
        """
        static_data = []

        for det in self.detector:
            fd_data = det.tdi_data.fd_data
            psd_data = det.tdi_data.fd_noise_power_density
            if (fd_data is None) or (psd_data is None):
                raise ValueError(
                    "The frequency-domain TDI data and noise power spectral density "
                    f"must be set for detector={det.name}."
                )
            for data_name, data_content in (
                ("fd_data", fd_data),
                ("fd_noise_power_density", psd_data),
            ):
                missing_chans = [
                    chan for chan in self.channels if chan not in data_content._fields
                ]
                if any(missing_chans):
                    raise ValueError(
                        f"Channels {missing_chans} are not contained in the "
                        f"`{data_name}` (channels {data_content._fields}) for "
                        f"detector={det.name}."
                    )

            # stack channels into (N_channels, N_freq) arrays, ordered as
            # self.channels, matching the channel order of the TDI combination
            # output; axis=0 is a contiguous concatenation of the per-channel
            # arrays (channel-first layout)
            data = jnp.stack([getattr(fd_data, chan) for chan in self.channels], axis=0)
            psd = jnp.stack([getattr(psd_data, chan) for chan in self.channels], axis=0)

            labels = det.tdi_combination.labels
            missing_chans = [chan for chan in self.channels if chan not in labels]
            if any(missing_chans):
                raise ValueError(
                    f"Channels {missing_chans} are not provided by the TDI "
                    f"combination model with labels {labels} of detector={det.name}."
                )
            channel_indices = tuple(labels.index(chan) for chan in self.channels)
            if channel_indices == tuple(range(len(labels))):
                # all channels are used in the native order, no slicing needed
                channel_indices = None

            det_data = DetectorData(
                data=data,
                psd=psd,
                df=det.tdi_data.data_info.delta_frequency,
                channel_indices=channel_indices,
            )
            static_data.append(det_data)

        self._static_data = tuple(static_data)

    def log_likelihood(self, params) -> float:
        """
        log-likelihood evaluation with jax.

        Parameters
        ----------
        params : tuple
            Namedtuple of parameters, including ``ecliptic_longitude``, ``ecliptic_latitude``,
            ``polarization``, ``coalescence_time`` for response computation, and parameters
            for waveform which depends on the waveform model.

        Returns
        -------
        float
            The value of log-likelihood, summed over all detectors and channels.

        Notes
        -----
        The waveform and the per-stage outputs are wrapped in
        ``jax.lax.optimization_barrier`` (which natively accepts pytrees)
        to prevent XLA from fusing the whole pipeline into a single kernel,
        which on the CPU backend produces much slower code (~6x slower
        without the barriers).
        """
        # The waveform is shared by all detectors (the frequency samples are
        # validated to match in ``__init__``), so it is computed once.
        wf = jax.lax.optimization_barrier(self.waveform(params))
        logl = 0.0
        for det, det_data in zip(self.detector, self._static_data):
            links = det.response_model._backend_impl.get_response_jit(
                wf,
                params.ecliptic_longitude,
                params.ecliptic_latitude,
                params.polarization,
                params.coalescence_time,
            )
            links = jax.lax.optimization_barrier(links)
            # tuple of per-channel (N_freq,) arrays, ordered as
            # det.tdi_combination.labels
            resps = det.tdi_combination._backend_impl.get_combination_jit(links)
            resps = jax.lax.optimization_barrier(resps)
            channel_indices = det_data.channel_indices
            if channel_indices is not None:
                # static indices, resolved at trace time
                resps = tuple(resps[i] for i in channel_indices)
            # stack channels into (N_channels, N_freq), axis=0 keeps each
            # channel contiguous
            resps = jnp.stack(resps, axis=0)

            logl += noise_weighted_inner_product(
                det_data.data, resps, det_data.psd, det_data.df
            )

        return logl
