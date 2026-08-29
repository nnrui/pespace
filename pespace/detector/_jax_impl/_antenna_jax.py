"""Computing detector single link response with jax."""

from __future__ import annotations
from collections import namedtuple
import weakref

import jax.numpy as jnp

from ..antenna import SingleLinkResponseModel
from ...utils.constants import *

PolarizationTuple = namedtuple("PolarizationTuple", ["plus", "cross"])


def get_polarization_tensor_ssb_jax(lam, beta, psi):
    """Compute GW polarization tensors in SSB frame (jax).

    See Also
    --------
    :func:`pespace.detector.antenna.get_polarization_tensor_ssb_taichi`
    """
    sin_lam, cos_lam = jnp.sin(lam), jnp.cos(lam)
    sin_beta, cos_beta = jnp.sin(beta), jnp.cos(beta)
    sin_psi, cos_psi = jnp.sin(psi), jnp.cos(psi)
    p = jnp.stack(
        [
            sin_lam * cos_psi - cos_lam * sin_beta * sin_psi,
            -cos_lam * cos_psi - sin_lam * sin_beta * sin_psi,
            cos_beta * sin_psi,
        ],
        axis=-1,
    )
    q = jnp.stack(
        [
            -sin_lam * sin_psi - cos_lam * sin_beta * cos_psi,
            cos_lam * sin_psi - sin_lam * sin_beta * cos_psi,
            cos_beta * cos_psi,
        ],
        axis=-1,
    )
    return PolarizationTuple(
        plus=jnp.outer(p, p) - jnp.outer(q, q),
        cross=jnp.outer(p, q) + jnp.outer(q, p),
    )


def get_gw_propagation_unit_vector_jax(lam, beta):
    """Compute the gravitational wave propagation unit vector (jax).

    See Also
    --------
    :func:`pespace.detector.antenna.get_gw_propagation_unit_vector_taichi`
    """
    return jnp.stack(
        [
            -jnp.cos(beta) * jnp.cos(lam),
            -jnp.cos(beta) * jnp.sin(lam),
            -jnp.sin(beta),
        ],
        axis=-1,
    )


class _FDResponseModelMarset2018Jax(SingleLinkResponseModel):
    """Jax backend for FDResponseModelMarset2018.

    Notes
    -----
    :meth:`update_single_link_response` is a wrapper that calls :meth:`get_response_jit`
    and stores the result onto the detector object. This mutates detector state and
    therefore cannot be jit-compiled. It is mainly used to maintain a consistent interface
    with the original taichi-based :class:`pespace.detector.antenna._FDResponseModelMarset2018Taichi`.

    :meth:`get_response_jit` reads static orbit model and frequency samples from ``self.detector``.
    It can be used in jit-able functions of jax via closure.
    """

    def init_single_link_response_model(self, detector: InterferometerAntenna) -> None:
        """Binding the instance of `InterferometerAntenna`."""
        self.detector = weakref.proxy(detector)

    def get_response_jit(self, waveform, lam, beta, psi, tc):
        """
        Compute single-link responses, jit-able via closure over ``self``.

        Parameters
        ----------
        waveform : dict
            Waveform dict. For a single-mode waveform: ``{"plus": hp,
            "cross": hc, "tf": tf}``; for waveforms with higher modes, each
            harmonic mode maps to such a dict, e.g. ``{"22": {...}, "21":
            {...}}``. ``hp`` and ``hc`` are complex arrays of length
            ``N_freq`` matching the detector frequency samples; ``tf`` is a
            real array of the same length.
        lam, beta, psi : float
            Sky location and polarization angle in radians.
        tc : float
            Coalescence time in seconds.

        Returns
        -------
        tuple of jnp.ndarray
            Six single-link responses, each of shape ``(N_freq,)``, ordered as
            (link12, link21, link23, link32, link31, link13).
        """
        orbit = self.detector.orbit_model
        freqs = self.detector.tdi_data.frequency_samples

        pol_tensor = get_polarization_tensor_ssb_jax(lam, beta, psi)  # matrix: 3*3
        prop_direc = get_gw_propagation_unit_vector_jax(lam, beta)  # vector: 3

        # single-mode waveform: {"plus": hp, "cross": hc, "tf": tf}; higher
        # modes: mode-keyed dict {"22": {...}, "21": {...}}.
        modes = (waveform,) if "plus" in waveform else tuple(waveform.values())

        def _response_mode(wf):
            hp, hc, tf = wf["plus"], wf["cross"], wf["tf"]  # each (N_freq,)

            tshift = jnp.exp(-1j * 2.0 * PI * freqs * tc)
            hp = hp * tshift
            hc = hc * tshift
            tf = tf + tc

            det_vecs = orbit.get_constellation_vectors(tf)  # 6 vectors of (N_freq, 3)
            arm_sec = orbit.arm_length_sec(tf)  # scalar, constant arm-length

            # antenna pattern functions n^T P n; the ellipsis keeps leading
            # batch dims when this function is vmapped over samples.
            n1_h_n1 = (
                jnp.einsum(
                    "...i,ij,...j->...", det_vecs.n1, pol_tensor.plus, det_vecs.n1
                )
                * hp
                + jnp.einsum(
                    "...i,ij,...j->...", det_vecs.n1, pol_tensor.cross, det_vecs.n1
                )
                * hc
            )
            n2_h_n2 = (
                jnp.einsum(
                    "...i,ij,...j->...", det_vecs.n2, pol_tensor.plus, det_vecs.n2
                )
                * hp
                + jnp.einsum(
                    "...i,ij,...j->...", det_vecs.n2, pol_tensor.cross, det_vecs.n2
                )
                * hc
            )
            n3_h_n3 = (
                jnp.einsum(
                    "...i,ij,...j->...", det_vecs.n3, pol_tensor.plus, det_vecs.n3
                )
                * hp
                + jnp.einsum(
                    "...i,ij,...j->...", det_vecs.n3, pol_tensor.cross, det_vecs.n3
                )
                * hc
            )

            k_n1 = jnp.einsum("i,...i->...", prop_direc, det_vecs.n1)
            k_n2 = jnp.einsum("i,...i->...", prop_direc, det_vecs.n2)
            k_n3 = jnp.einsum("i,...i->...", prop_direc, det_vecs.n3)
            k_x1_x2 = jnp.einsum("i,...i->...", prop_direc, det_vecs.x1 + det_vecs.x2)
            k_x2_x3 = jnp.einsum("i,...i->...", prop_direc, det_vecs.x2 + det_vecs.x3)
            k_x3_x1 = jnp.einsum("i,...i->...", prop_direc, det_vecs.x3 + det_vecs.x1)

            f_L = freqs * arm_sec
            # Note jnp.sinc is the normalized sinc: sin(pi*x)/(pi*x)!
            sinc32 = jnp.sinc(f_L * (1.0 - k_n1))
            sinc23 = jnp.sinc(f_L * (1.0 + k_n1))
            sinc13 = jnp.sinc(f_L * (1.0 - k_n2))
            sinc31 = jnp.sinc(f_L * (1.0 + k_n2))
            sinc21 = jnp.sinc(f_L * (1.0 - k_n3))
            sinc12 = jnp.sinc(f_L * (1.0 + k_n3))

            common_exp = -PI * freqs * 1j  # complex
            exp12 = jnp.exp(common_exp * (arm_sec + k_x1_x2))
            exp23 = jnp.exp(common_exp * (arm_sec + k_x2_x3))
            exp31 = jnp.exp(common_exp * (arm_sec + k_x3_x1))

            prefactor = -PI * f_L * 1j
            link12 = prefactor * sinc12 * n3_h_n3 * exp12
            link21 = prefactor * sinc21 * n3_h_n3 * exp12
            link23 = prefactor * sinc23 * n1_h_n1 * exp23
            link32 = prefactor * sinc32 * n1_h_n1 * exp23
            link31 = prefactor * sinc31 * n2_h_n2 * exp31
            link13 = prefactor * sinc13 * n2_h_n2 * exp31

            return link12, link21, link23, link32, link31, link13

        links = _response_mode(modes[0])
        for wf_mode in modes[1:]:
            links_mode = _response_mode(wf_mode)
            links = tuple(a + b for a, b in zip(links, links_mode))
        return links

    def update_single_link_response(self, waveform, lam, beta, psi, tc) -> None:
        """Compute single link response and update the detector state.

        This is a wrapper around :meth:`get_response_jit` and will writes
        the result to ``self.detector.single_link_response``.

        Parameters
        ----------
        waveform : dict
            Waveform dict, see :meth:`get_response_jit` for the structure.
        lam, beta, psi : float
            Sky location and polarization angle in radians.
        tc : float
            Coalescence time in seconds.
        """
        links = self.get_response_jit(
            waveform,
            lam,
            beta,
            psi,
            tc,
        )
        DataContent = namedtuple(
            "DataContent", ("link12", "link21", "link23", "link32", "link31", "link13")
        )
        self.detector.single_link_response = DataContent(*links)
