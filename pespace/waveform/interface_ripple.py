"""
Interface to ripple waveforms.

The frequency-domain waveform generators in ripple used here are implemented
with jax in a non-official fork, so the generated waveforms are jit-able and
grad-able, which makes them suitable for likelihood computations with jax
backends (see
:class:`pespace.inference.likelihood_jax.WhittleLikelihoodJax`).

ripple is not a default dependency of pespace. The import is attempted when
this module is loaded; if ripple is not installed, an ``ImportError``
reminding the user to install the non-official ripple from
``_RIPPLE_INSTALL_URL`` is raised immediately.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp

#: URL of the repository of the non-official ripple (jax-based).
_RIPPLE_INSTALL_URL = "https://github.com/nnrui/ripple"


try:
    from ripplegw.waveforms.IMRPhenomXAS import gen_IMRPhenomXAS_hphctf
    from ripplegw.waveforms.IMRPhenomXHM import gen_IMRPhenomXHM_hphctf_modes
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


class IMRPhenomXHM(WaveformRippleBase):
    """
    Interface class to get the IMRPhenomXHM waveform from ripple.

    The waveform is returned mode by mode, since the detector response is
    computed for each mode separately from its own time-frequency relation
    ``tf`` (see :mod:`pespace.detector._jax_impl._antenna_jax`).

    Parameters
    ----------
    frequencies : array-like
        Frequency samples in Hz at which the waveform is evaluated.
    reference_frequency : float
        Reference frequency in Hz for the waveform generation.
    modes : tuple of str, optional
        Higher modes to include besides the (2, 2) mode, by default
        ``("21", "33", "32", "44")``.

    Notes
    -----
    Unlike :class:`IMRPhenomXAS`, which requires the chirp mass and the
    symmetric mass ratio, this template takes the component masses
    ``mass_1`` and ``mass_2`` directly, so that no mass conversion is
    performed inside the interface.
    """

    def __init__(
        self, frequencies, reference_frequency: float, modes=("21", "33", "32", "44")
    ):
        super().__init__(frequencies, reference_frequency)
        self.modes = tuple(modes)

    @property
    def required_params(self):
        """Names of the parameters required by IMRPhenomXHM."""
        return (
            "mass_1",
            "mass_2",
            "chi_1",
            "chi_2",
            "luminosity_distance",
            "reference_phase",
            "inclination",
        )

    def get_waveform(self, params):
        """
        Evaluate the IMRPhenomXHM waveform from ripple.

        Parameters
        ----------
        params : namedtuple
            Namedtuple containing the parameters listed in
            :attr:`required_params`.

        Returns
        -------
        dict
            Mode-keyed waveform dict, e.g.
            ``{"22": {"plus": hp, "cross": hc, "tf": tf}, ...}``, one entry
            per mode: the (2, 2) mode plus each mode in :attr:`modes`. Each
            entry holds the complex polarizations ``hp``/``hc`` and the real
            time-frequency relation ``tf`` of that mode.
        """
        # the coalescence time shift is applied in the response computation,
        # so it is fixed to zero here
        params_array = jnp.array(
            [
                params.mass_1,
                params.mass_2,
                params.chi_1,
                params.chi_2,
                params.luminosity_distance,
                0.0,
                params.reference_phase,
                params.inclination,
            ]
        )
        return gen_IMRPhenomXHM_hphctf_modes(
            self.frequencies, params_array, self.reference_frequency, modes=self.modes
        )
