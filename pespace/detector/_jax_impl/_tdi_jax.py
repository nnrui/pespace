"""Computing the TDI combination with jax"""

from collections import namedtuple
import weakref

import jax.numpy as jnp

from ..tdi import TDICombinationModel
from ...utils.constants import *


class _FDMichelsonConstantEqualArmJax(TDICombinationModel):
    """Jax backend for FDMichelsonConstantEqualArm.

    Notes
    -----
    Similar to :class:`._antenna_jax._FDResponseModelMarset2018Jax`,
    :meth:`update_tdi_response` is a wrapper that calls :meth:`get_combination_jit`
    and stores the result onto the detector object. This mutates detector state and
    therefore cannot be jit-compiled. It is mainly used to maintain a consistent
    interface with the original taichi-based
    :class:`pespace.detector.tdi._FDMichelsonConstantEqualArmTaichi`.

    :meth:`get_combination_jit` reads static ``tdi_model.orthogonal`` and
    ``_cached_data`` from ``self``, it need to be captured via closure for jit.
    """

    def __init__(self, tdi_model) -> None:
        """Bind the public interface class holding the static configuration."""
        self.tdi_model = tdi_model

    def init_tdi_combination_model(self, detector) -> None:
        """Binding the instance of `InterferometerAntenna`."""
        self.detector = weakref.proxy(detector)
        data_zero = jnp.zeros(
            (
                len(self.tdi_model.labels),
                self.detector.tdi_data.data_info.frequency_series_length,
            ),
            dtype=complex,
        )
        DataContent = namedtuple("DataContent", self.tdi_model.labels)
        self.detector.tdi_response = DataContent(*data_zero)
        self._set_cached_data()

    def _set_cached_data(self):
        """
        Compute and cache frequency-dependent prefactors and delay factors.

        This mirrors the taichi implementation of
        :meth:`_FDMichelsonConstantEqualArmTaichi._set_cached_field`.
        """
        CachedData = namedtuple("CachedData", ["prefactor", "delay_factor"])

        freqs = self.detector.tdi_data.frequency_samples
        arm_sec = self.detector.orbit_model.arm_length_sec(0.0)
        # delay factor exp(-i 2 pi f L)
        delay_factor = jnp.exp(-2.0 * PI * freqs * arm_sec * 1j)

        gen = self.tdi_model.generation
        if gen == "1.5":
            prefactor = 1.0 - delay_factor**2
        elif gen == "2.0":
            prefactor = 1.0 - delay_factor**2 - delay_factor**4 + delay_factor**6
        else:
            raise ValueError(f"Invalid tdi generation {gen}.")

        self._cached_data = CachedData(prefactor=prefactor, delay_factor=delay_factor)

    def get_combination_jit(self, single_link_response):
        """
        Compute TDI combination from single-link responses.

        Orthogonal flag and cached prefactors are read from ``self`` and need to be
        captured by closure for JIT compilation.

        Parameters
        ----------
        single_link_response : tuple of jnp.ndarray
            Six single-link responses, each of shape ``(N_freq,)``, ordered as
            (link12, link21, link23, link32, link31, link13).

        Returns
        -------
        tuple of jnp.ndarray
            Three TDI combination channels, each of shape ``(N_freq,)``. The
            channel order corresponds to ``('A', 'E', 'T')`` if ``orthogonal``
            is True, otherwise ``('X', 'Y', 'Z')``.
        """
        orthogonal = self.tdi_model.orthogonal
        prefactor, delay_factor = self._cached_data

        link12, link21, link23, link32, link31, link13 = single_link_response
        X = prefactor * (link13 - link12 + delay_factor * (link31 - link21))
        Y = prefactor * (link21 - link23 + delay_factor * (link12 - link32))
        Z = prefactor * (link32 - link31 + delay_factor * (link23 - link13))
        if orthogonal:
            A = (Z - X) / jnp.sqrt(2.0)
            E = (X - 2.0 * Y + Z) / jnp.sqrt(6.0)
            T = (X + Y + Z) / jnp.sqrt(3.0)
            return A, E, T
        return X, Y, Z

    def update_tdi_response(self) -> None:
        """
        Compute TDI response and update the detector state.

        This is a wrapper around :meth:`get_combination_jit`, and writes the
        result to ``self.detector.tdi_response`` as a NamedTuple keyed by the
        channel labels of the TDI combination.

        It mutates detector state and therefore cannot be jit-compiled, used mainly to
        maintain a consistent interface with the previous taichi implementation.
        """
        tdi_response = self.get_combination_jit(self.detector.single_link_response)
        DataContent = namedtuple("DataContent", self.tdi_model.labels)
        self.detector.tdi_response = DataContent(*tdi_response)
