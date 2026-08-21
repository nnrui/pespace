"""
Interface to ripple waveforms.

The frequency-domain waveform generators in ripple used here are implemented
with jax in a non-official fork, so the generated waveforms are jit-able and
grad-able, which makes them suitable for likelihood computations with jax
backends (see
:class:`pespace.inference.interface_jax.WhittleLikelihoodJax`).

ripple is not a default dependency of pespace. It is imported lazily when a
waveform is requested; if it is not installed, a :class:`RippleImportError`
reminding the user to install the non-official ripple from
``_RIPPLE_INSTALL_URL`` is raised.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp

#: URL of the repository of the non-official ripple (jax-based).
#: TODO: replace with the real repository URL.
_RIPPLE_INSTALL_URL = "https://github.com/xxx"


class RippleImportError(ImportError):
    """
    Raised when ripple is not installed but a waveform is requested.

    The default message reminds the user to install the non-official ripple
    from ``_RIPPLE_INSTALL_URL``.
    """

    _message = (
        "ripple is not installed by default. If waveforms from ripple "
        "are needed, please install the non-official ripple manually "
        f"from {_RIPPLE_INSTALL_URL}."
    )

    def __init__(self):
        super().__init__(self._message)


class WaveformRippleBase(ABC):
    """
    Abstract base class of the ripple waveform interfaces.

    Concrete waveform templates should subclass this class, implement the
    :attr:`required_params` property and the :meth:`get_waveform` method, and
    are then callable through ``__call__``.

    Parameters
    ----------
    frequencies : array-like
        Frequency samples in Hz at which the waveform is evaluated.
    reference_frequency : float
        Reference frequency in Hz for the waveform generation.

    Notes
    -----
    The frequency-domain waveform returned by concrete templates has the AoS
    layout of shape ``(N_freq, 3)`` stacking ``(h_plus, h_cross, t_f)`` per
    frequency bin. The coalescence time shift is usually applied in the
    detector response computation; in that case the coalescence time is fixed
    to zero inside the waveform computation.
    """

    def __init__(self, frequencies, reference_frequency: float):
        self.frequencies = jnp.asarray(frequencies)
        self.reference_frequency = reference_frequency

    @property
    @abstractmethod
    def required_params(self) -> tuple[str, ...]:
        """
        Names of the parameters required by the waveform template.

        Returns
        -------
        tuple of str
            Required parameter names.
        """

    @abstractmethod
    def get_waveform(self, params) -> jnp.ndarray:
        """
        Evaluate the ripple waveform for the given parameters.

        Parameters
        ----------
        params : namedtuple
            Namedtuple containing the parameters listed in
            :attr:`required_params`.

        Returns
        -------
        jnp.ndarray
            Stacked waveform of shape ``(N_freq, 3)`` with
            ``(h_plus, h_cross, t_f)`` per frequency bin.

        Raises
        ------
        RippleImportError
            If ripple is not installed.
        """

    def __call__(self, params):
        """Evaluate the ripple waveform for the given parameters.

        Shortcut for :meth:`get_waveform`.
        """
        return self.get_waveform(params)


class IMRPhenomXAS(WaveformRippleBase):
    """
    Interface class to get the IMRPhenomXAS waveform from ripple.

    Parameters
    ----------
    frequencies : array-like
        Frequency samples in Hz at which the waveform is evaluated.
    reference_frequency : float
        Reference frequency in Hz for the waveform generation.
    """

    @property
    def required_params(self):
        """Names of the parameters required by IMRPhenomXAS."""
        return (
            "chirp_mass",
            "symmetric_mass_ratio",
            "chi_1",
            "chi_2",
            "luminosity_distance",
            "reference_phase",
            "inclination",
        )

    def get_waveform(self, params):
        """
        Evaluate the IMRPhenomXAS waveform from ripple.

        Parameters
        ----------
        params : namedtuple
            Namedtuple containing the parameters listed in
            :attr:`required_params`.

        Returns
        -------
        jnp.ndarray
            Stacked waveform of shape ``(N_freq, 3)`` with
            ``(h_plus, h_cross, t_f)`` per frequency bin.
        """
        try:
            from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphctf
        except ImportError:
            raise RippleImportError

        # the coalescence time shift is applied in the response computation,
        # so it is fixed to zero here
        params_array = jnp.array(
            [
                params.chirp_mass,
                params.symmetric_mass_ratio,
                params.chi_1,
                params.chi_2,
                params.luminosity_distance,
                0.0,
                params.reference_phase,
                params.inclination,
            ]
        )
        hp, hc, tf = gen_IMRPhenomXAS_hphctf(
            self.frequencies, params_array, self.reference_frequency
        )
        return jnp.stack([hp, hc, tf], axis=-1)
