"""Computing detector single link response with jax."""

from collections import namedtuple
import weakref

import jax
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
    with the original taichi-based :class:`SingleLinkResponseModel`.

    :meth:`get_response_jit` is a pure jax static method. All required data is passed-in
    as arguments, without side effects.
    """

    def init_single_link_response_model(self, detector: InterferometerAntenna) -> None:
        """Binding the instance of `InterferometerAntenna`."""
        self.detector = weakref.proxy(detector)

    @staticmethod
    def get_response_jit(orbit, freqs, waveform, lam, beta, psi, tc):
        """Compute single-link responses (pure function, jit-able).

        All required data is passed-in explicitly as arguments; no external state is
        read or written.

        Parameters
        ----------
        orbit : OrbitModel with jax-backend
            Jax-based orbit model providing constellation vectors and arm lengths.
        freqs : jnp.ndarray
            Frequency samples in Hz, shape ``(N_freqs,)``.
        waveform : jnp.ndarray
            Waveform array; shape ``(N_freqs, 3)``, (hp, hc, tf) per bin for single mode,
            or ``(N_freqs, N_modes, 3)`` for higher modes.
        lam, beta, psi : float
            Sky location and polarization angle in radians.
        tc : float
            Coalescence time in seconds.

        Returns
        -------
        jnp.ndarray
            Single-link responses, shape ``(N_freq, 6)``,
            with (link12, link21, link23, link32, link31, link13) per bin.
        """
        pol_tensor = get_polarization_tensor_ssb_jax(lam, beta, psi)  # matrix: 3*3
        prop_direc = get_gw_propagation_unit_vector_jax(lam, beta)  # vector: 3

        def _response_at_fi(wf, fi):
            hp, hc, tf = wf
            tshift = jnp.exp(-1j * 2.0 * PI * fi * tc)
            hp *= tshift
            hc *= tshift
            tf += tc

            det_vecs = orbit.get_constellation_vectors(tf)
            arm_sec = orbit.arm_length_sec(tf)
            # n1: unit vector of 2 -> 3
            n1_h_n1 = (
                det_vecs.n1 @ pol_tensor.plus @ det_vecs.n1 * hp
                + det_vecs.n1 @ pol_tensor.cross @ det_vecs.n1 * hc
            )
            # n2: unit vector of 3 -> 1
            n2_h_n2 = (
                det_vecs.n2 @ pol_tensor.plus @ det_vecs.n2 * hp
                + det_vecs.n2 @ pol_tensor.cross @ det_vecs.n2 * hc
            )
            # n3: unit vector of 1 -> 2
            n3_h_n3 = (
                det_vecs.n3 @ pol_tensor.plus @ det_vecs.n3 * hp
                + det_vecs.n3 @ pol_tensor.cross @ det_vecs.n3 * hc
            )

            k_n1 = prop_direc @ det_vecs.n1  # scalar
            k_n2 = prop_direc @ det_vecs.n2  # scalar
            k_n3 = prop_direc @ det_vecs.n3  # scalar

            k_x1_x2 = prop_direc @ (det_vecs.x1 + det_vecs.x2)  # scalar
            k_x2_x3 = prop_direc @ (det_vecs.x2 + det_vecs.x3)  # scalar
            k_x3_x1 = prop_direc @ (det_vecs.x3 + det_vecs.x1)  # scalar

            pi_f_L = PI * fi * arm_sec  # scalar
            sinc32 = jnp.sinc(pi_f_L * (1.0 - k_n1))  # scalar
            sinc23 = jnp.sinc(pi_f_L * (1.0 + k_n1))  # scalar
            sinc13 = jnp.sinc(pi_f_L * (1.0 - k_n2))  # scalar
            sinc31 = jnp.sinc(pi_f_L * (1.0 + k_n2))  # scalar
            sinc21 = jnp.sinc(pi_f_L * (1.0 - k_n3))  # scalar
            sinc12 = jnp.sinc(pi_f_L * (1.0 + k_n3))  # scalar

            common_exp = -PI * fi * 1j  # complex
            exp12 = jnp.exp(common_exp * (arm_sec + k_x1_x2))  # complex
            exp23 = jnp.exp(common_exp * (arm_sec + k_x2_x3))  # complex
            exp31 = jnp.exp(common_exp * (arm_sec + k_x3_x1))  # complex

            link12 = sinc12 * n3_h_n3 * exp12
            link21 = sinc21 * n3_h_n3 * exp12
            link23 = sinc23 * n1_h_n1 * exp23
            link32 = sinc32 * n1_h_n1 * exp23
            link31 = sinc31 * n2_h_n2 * exp31
            link13 = sinc13 * n2_h_n2 * exp31

            response = jnp.stack(
                [link12, link21, link23, link32, link31, link13], axis=-1
            )
            prefactor = -pi_f_L * 1j
            response *= prefactor
            return response

        def _response_at_fi_modes(wf_modes, fi):
            mode_num, _ = wf_modes.shape
            response = jnp.zeros(6, dtype=complex)
            for i in range(mode_num):
                response_mode = _response_at_fi(wf_modes[i], fi)
                response += response_mode
            return response

        if waveform.ndim == 2:
            response = jax.vmap(_response_at_fi)(waveform, freqs)
        elif waveform.ndim == 3:
            response = jax.vmap(_response_at_fi_modes)(waveform, freqs)
        else:
            raise ValueError("Invalid dimension of waveform.")

        return response

    def update_single_link_response(self, waveform, lam, beta, psi, tc) -> None:
        """Compute single link response and update the detector state.

        This is a wrapper around :meth:`get_response_jit` and will writes
        the result to ``self.detector.single_link_response``.

        Parameters
        ----------
        waveform : jnp.ndarray
            Waveform array; shape ``(N_freqs, 3)``, (hp, hc, tf) per bin for single mode,
            or ``(N_freqs, N_modes, 3)`` for higher modes.
        lam, beta, psi : float
            Sky location and polarization angle in radians.
        tc : float
            Coalescence time in seconds.
        """
        self.detector.single_link_response = self.get_response_jit(
            self.detector.orbit_model,
            self.detector.tdi_data.frequency_samples,
            waveform,
            lam,
            beta,
            psi,
            tc,
        )
