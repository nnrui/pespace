"""Taichi backend of the Whittle likelihood kernel.

This module contains the taichi kernel for the Whittle likelihood
computation, used by
:class:`~pespace.inference.interface_bilby.LikelihoodBilbyInterfaceTaichi`.

Important: this module must *not* use ``from __future__ import annotations``
(PEP 563). Taichi parses kernel annotations at decoration time and requires
the actual annotation objects (e.g., ``ti.template()``); with PEP 563 all
annotations become strings and the kernel definition fails with a
``TaichiSyntaxError``.
"""

import taichi as ti


@ti.kernel
def compute_whittle_likelihood_ti(
    channels: ti.template(),
    observed_data: ti.template(),
    response_data: ti.template(),
    psd: ti.template(),
    df: float,
) -> float:
    """Compute the Whittle likelihood.

    Parameters
    ----------
    channels : ti.template()
        Specifying the TDI channels to iterate over.
        Used with ``ti.static()`` for compile-time loop unrolling.
    observed_data : ti.template()
        ``taichi.field`` containing the observed frequency-domain data for each channel.
        Expected to be a field of ``ti.types.vector(2, float)`` indexed by frequency bin
        and channel.
    response_data : ti.template()
        ``taichi.field`` containing detector responses for each channel, having the same
        structure as ``observed_data``.
    psd : ti.template()
        ``taichi.field`` containing the noise power spectral density for each channel.
        Expected to be a real-valued field indexed by frequency bin and channel.
    df : float
        Spacing between frequency bins in Hz.

    Returns
    -------
    float
        The log-likelihood value computed using the Whittle approximation.

    Notes
    -----

    - Uses Array-of-Structures (AoS) layout for ``StructField``, with the channel
      loop placed inside the frequency loop for optimal memory accessing
    - Atomic operations are used to accumulate the log-likelihood to ensure
      thread safety in parallel execution
    """
    log_l = 0.0
    for i in observed_data:
        inner_product = 0.0
        for chan in ti.static(channels):
            # AoS is used for StructField, placing the loop of channels inside.
            inner_product += (
                observed_data[i][chan] - response_data[i][chan]
            ).norm_sqr() / psd[i][chan]
        ti.atomic_add(log_l, inner_product)
    log_l *= -2 * df

    return log_l
