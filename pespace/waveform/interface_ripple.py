"""
Interface to ripple waveforms.

The frequency-domain waveform generators in ripple used here are implemented
with jax in a non-official fork, so the generated waveforms are jit-able and
grad-able, which makes them suitable for likelihood computations with jax
backends (see
:class:`pespace.inference.interface_jax.WhittleLikelihoodJax`).

ripple is not a default dependency of pespace. The import is attempted when
this module is loaded; if ripple is not installed, an ``ImportError``
reminding the user to install the non-official ripple from
``_RIPPLE_INSTALL_URL`` is raised immediately.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp

#: URL of the repository of the non-official ripple (jax-based).
#: TODO: replace with the real repository URL.
_RIPPLE_INSTALL_URL = "https://github.com/xxx"


try:
    from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphctf
except ImportError as err:
    raise ImportError(
        "ripple is not installed by default. If waveforms from ripple "
        "are needed, please install the non-official ripple manually "
        f"from {_RIPPLE_INSTALL_URL}."
    ) from err


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
    The frequency-domain waveform returned by concrete templates is a dict:
    ``{"plus": hp, "cross": hc, "tf": tf}`` for a single-mode waveform, with
    complex ``hp``/``hc`` and real ``tf`` of shape ``(N_freq,)``; for waveforms
    with higher modes, each harmonic mode maps to such a dict, e.g.
    ``{"22": {...}, "21": {...}}``. The coalescence time shift is usually
    applied in the detector response computation; in that case the coalescence
    time is fixed to zero inside the waveform computation.
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
    def get_waveform(self, params) -> dict:
        """
        Evaluate the ripple waveform for the given parameters.

        Parameters
        ----------
        params : namedtuple
            Namedtuple containing the parameters listed in
            :attr:`required_params`.

        Returns
        -------
        dict
            Waveform dict: ``{"plus": hp, "cross": hc, "tf": tf}`` for a
            single-mode waveform, or a mode-keyed dict
            ``{"22": {...}, "21": {...}}`` for higher modes.
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
        dict
            Waveform dict: ``{"plus": hp, "cross": hc, "tf": tf}`` for a
            single-mode waveform, or a mode-keyed dict
            ``{"22": {...}, "21": {...}}`` for higher modes.
        """
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
        return {"plus": hp, "cross": hc, "tf": tf}
