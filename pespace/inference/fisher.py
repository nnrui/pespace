"""Fisher matrix computation for the Whittle likelihood.

In the high-SNR regime the likelihood can be expanded as a series in
``1 / rho``; keeping the leading order, the posterior distribution with
uniform priors is approximated by a multivariate Gaussian distribution with
means given by the fiducial parameters and covariance matrix given by the
inverse of the Fisher information matrix (Vallisneri 2008),

.. math::

    F_{ij} = \\left\\langle \\frac{\\partial \\tilde{s}}{\\partial
    \\theta_i} \\middle| \\frac{\\partial \\tilde{s}}{\\partial \\theta_j}
    \\right\\rangle,

where the noise-weighted inner product is

.. math::

    \\langle a | b \\rangle = 4 \\mathrm{Re} \\sum_k
    \\frac{a^*(f_k) b(f_k)}{S_n(f_k) T}.
"""

import jax
import jax.numpy as jnp


def fisher_matrix(likelihood, fiducial_params, batch_size=None):
    """
    Compute the Fisher information matrix at one or more fiducial parameter
    sets.

    The derivatives of the model response with respect to all parameters of
    the likelihood are computed with forward-mode automatic differentiation,
    and the Fisher matrix is assembled from the noise-weighted inner products
    of the derivatives, summed over all detectors and requested channels,

    ``F_ij = 4 * df * Re sum_{c, f} conj(ds_i(f)) * ds_j(f) / psd_c(f)``.

    A single parameter set returns a single matrix; a batch of parameter sets
    is evaluated with ``vmap`` (optionally split into chunks by
    ``batch_size``), returning one matrix per set.

    Parameters
    ----------
    likelihood : WhittleLikelihoodJax
        Likelihood instance defining the waveform model, the detectors with
        their noise power spectral densities, and the requested TDI channels.
        The observed data of the detectors is not used: the Fisher matrix only
        depends on the model and the noise power spectral densities.
    fiducial_params : dict
        Fiducial parameters at which the Fisher matrix is evaluated, e.g. the
        injected parameters of the simulated signal. Must contain all
        parameters of ``likelihood.likelihood_param_type`` (the waveform
        parameters followed by the response parameters); additional entries
        are ignored. A single parameter set is given by scalar values; a batch
        of parameter sets is given by 1-D arrays whose (common) leading axis
        enumerates the sets, with scalar entries broadcast over the batch.
    batch_size : int or None, optional
        Number of parameter sets evaluated at once within ``vmap``. ``None``
        (default) evaluates the whole batch with a single ``vmap`` call; an
        integer evaluates the batch in chunks of that size, bounding the
        memory footprint of large batches. Ignored for a single parameter set.

    Returns
    -------
    jnp.ndarray
        Fisher matrix of shape ``(N_params, N_params)`` for a single parameter
        set, or ``(N_batch, N_params, N_params)`` for a batch of ``N_batch``
        parameter sets. ``N_params`` is the number of parameters of
        ``likelihood.likelihood_param_type``, and the rows and columns follow
        the order of its fields.

    Notes
    -----
    - The posterior distribution is approximated by a multivariate Gaussian
      with mean ``fiducial_params`` and covariance matrix
      ``jnp.linalg.inv(fisher_matrix(...))``.
    - When the observed data are noiseless and equal to the response of
      ``fiducial_params``, the Fisher matrix equals minus the Hessian of
      :meth:`~pespace.inference.likelihood_jax.WhittleLikelihoodJax.log_likelihood`
      evaluated at ``fiducial_params``.
    """
    # the channel sum assumes mutually independent A, E, T channels
    if not set(likelihood.channels) <= {"A", "E", "T"}:
        raise ValueError(
            "fisher_matrix requires the independent orthogonal channels "
            f"('A', 'E', 'T'), but got the channels {tuple(likelihood.channels)}."
        )

    param_names = tuple(likelihood.likelihood_param_type._fields)
    param_values = {name: jnp.asarray(fiducial_params[name]) for name in param_names}

    def fisher_at(params_dict):
        """Fisher matrix at a single parameter set."""
        params = likelihood.likelihood_param_type(**params_dict)
        fisher = jnp.zeros((len(param_names), len(param_names)))

        for detector, det_data in zip(likelihood.detector, likelihood._static_data):

            def response(params):
                """Model TDI response, stacked as ``(N_channels, N_freq)``.

                Mirrors the response pipeline of ``log_likelihood``, including
                the ``jax.lax.optimization_barrier`` calls which prevent XLA
                from fusing the pipeline into a single (much slower) kernel.
                """
                waveform = jax.lax.optimization_barrier(likelihood.waveform(params))
                links = detector.response_model._backend_impl.get_response_jit(
                    waveform,
                    params.ecliptic_longitude,
                    params.ecliptic_latitude,
                    params.polarization,
                    params.coalescence_time,
                )
                links = jax.lax.optimization_barrier(links)
                responses = detector.tdi_combination._backend_impl.get_combination_jit(
                    links
                )
                responses = jax.lax.optimization_barrier(responses)
                if det_data.channel_indices is not None:
                    responses = tuple(
                        responses[index] for index in det_data.channel_indices
                    )
                return jnp.stack(responses, axis=0)

            # jacfwd returns a namedtuple over the parameters, each leaf being
            # d(response)/d(param) with shape (N_channels, N_freq)
            jacobian = jax.jacfwd(response)(params)
            derivatives = jnp.stack(
                [getattr(jacobian, name) for name in param_names], axis=0
            )  # shape (N_params, N_channels, N_freq)

            # F_ij = 4 df * Re sum_{c, f} conj(ds_i) * ds_j / psd
            fisher += (
                4.0
                * det_data.df
                * jnp.real(
                    jnp.einsum(
                        "icf,jcf,cf->ij",
                        derivatives.conj(),
                        derivatives,
                        1.0 / det_data.psd,
                    )
                )
            )

        return fisher

    batch_shape = jnp.broadcast_shapes(
        *(value.shape for value in param_values.values())
    )
    if batch_shape == ():
        # a single parameter set
        return fisher_at(param_values)

    param_values = {
        name: jnp.broadcast_to(value, batch_shape)
        for name, value in param_values.items()
    }
    if batch_size is None:
        return jax.vmap(fisher_at)(param_values)
    return jax.lax.map(fisher_at, param_values, batch_size=batch_size)
