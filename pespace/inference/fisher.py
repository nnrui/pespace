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


def sample_from_fisher(fisher, num, rkey_seed=0):
    """
    Draw random samples from the Gaussian approximation ``N(0, F^{-1})``.

    The high-SNR posterior approximation associated with a Fisher matrix
    ``F`` is a multivariate Gaussian whose mean is given by the fiducial
    parameters and whose covariance matrix is ``F^{-1}`` (see
    :func:`fisher_matrix`). This function draws zero-mean samples from that
    Gaussian; the fiducial parameter values must be added by the caller.

    The samples are obtained without forming or inverting the covariance
    matrix. The Fisher matrix is first rescaled to unit diagonal with the
    dimensionless matrix ``F' = F / (sqrt(diag(F)) outer sqrt(diag(F)))``,
    which removes the disparity of the parameter units (for a representative
    mixed-unit gravitational-wave case the condition number drops from
    ~1e19 to ~1e8, leaving a comfortable margin within float64). The scaled
    matrix is factorized with a Cholesky decomposition, ``F' = L L^T``, and
    each sample is computed by solving the triangular system ``L^T x' = z``
    for standard normal ``z``, so that ``Cov(x') = L^{-T} L^{-1} = F'^{-1}``;
    the samples are then rescaled back to the original units. The solve is
    backward stable and avoids both the explicit matrix inversion and the
    loss of symmetry and positive definiteness that make a naive
    ``inv(F)``-then-sample route unreliable at such condition numbers.

    Parameters
    ----------
    fisher : array_like, shape (N, N)
        Fisher information matrix, symmetric positive definite, with rows
        and columns ordered as the parameters of the likelihood.
    num : int
        Number of samples to draw.
    rkey_seed : int, optional
        Seed used to draw the standard normal variates. Defaults to 0, so
        repeated calls return identical samples.

    Returns
    -------
    jnp.ndarray, shape (num, N)
        Zero-mean samples with covariance matrix ``fisher^{-1}``, in the
        same parameter order and units as ``fisher``.

    Notes
    -----
    - Double precision should be enabled (``jax_enable_x64``); in float32
      the diagonal rescaling cannot compensate the extreme condition number
      of the mixed-unit Fisher matrix.
    - The function is compatible with ``jax.jit``; ``jax.vmap`` over the
      Fisher matrices (with a common seed) draws samples for several
      parameter sets at once.
    """
    key = jax.random.key(int(rkey_seed))

    fisher = jnp.asarray(fisher)

    # Jacobi (diagonal) preconditioning: a unit diagonal removes the unit
    # disparity of the parameters so that the factorization remains well
    # posed numerically.
    scale = jnp.sqrt(jnp.diag(fisher))
    fisher_scaled = fisher / jnp.outer(scale, scale)

    # F' = L L^T, and x' = L^{-T} z has covariance F'^{-1}; the triangular
    # system L^T x' = z is solved directly, without forming any inverse.
    cholesky = jnp.linalg.cholesky(fisher_scaled)
    normals = jax.random.normal(key, (num, fisher.shape[0]))
    samples = jnp.linalg.solve(cholesky.T, normals.T).T

    # back to the original units: x = x' / scale gives Cov(x) = F^{-1}
    return samples / scale


def wrap_angle_params(samples):
    """
    Wrap the angle parameters of a sample set into finite canonical ranges.

    The following entries of ``samples`` are wrapped if present; all other
    entries are left unchanged:

    - ``ecliptic_longitude``: wrapped modulo ``2*pi`` into ``[0, 2*pi)``;
    - ``ecliptic_latitude``: folded by reflection about the boundaries
      ``-pi/2`` and ``pi/2`` into ``[-pi/2, pi/2]``;
    - ``inclination``: folded by reflection about the boundaries ``0`` and
      ``pi`` into ``[0, pi]``;
    - ``polarization``: wrapped modulo ``pi`` into ``[0, pi)`` (the detector
      response depends on the polarization angle through its double angle,
      so ``pi`` is its fundamental period);
    - ``reference_phase``: wrapped modulo ``2*pi`` into ``[0, 2*pi)``.

    These are the finite ranges conventionally used for the angle parameters
    of the space-borne detector response; they match the sampling ranges
    adopted in the example scripts.

    Parameters
    ----------
    samples : mapping
        Mapping from parameter names to sample values, e.g. a ``dict`` of
        arrays or a ``pandas.DataFrame``; values may be scalars or arrays of
        any array library (Python, NumPy or JAX).

    Returns
    -------
    mapping
        A copy of ``samples`` with the angle parameters restricted to their
        finite ranges; neither the input container nor its arrays are
        modified. The copy is shallow: the wrapped entries are replaced by
        newly created arrays, while the remaining entries are shared with
        the input (a deep copy is not needed, since no array is modified in
        place).

    Notes
    -----
    - The wrapping is exact for the periodic angles (``ecliptic_longitude``,
      ``polarization`` and ``reference_phase``) and is a reflection fold for
      the bounded angles (``ecliptic_latitude`` and ``inclination``).
    - In this likelihood the ``reference_phase`` is periodic with period
      ``pi`` rather than ``2*pi`` (the strain depends on ``2 *
      reference_phase``), so the two wells at ``phi`` and ``phi + pi`` are
      exactly symmetric; the wrapping above keeps the conventional
      ``[0, 2*pi)`` range, and the wells can be identified with
      ``reference_phase % pi`` when needed.
    - The wrapping is compatible with ``jax.jit``; the reflection folds use
      ``jnp.abs``, so those results are JAX arrays even for non-JAX inputs.
    """
    try:
        keys = set(samples.keys())
    except AttributeError as error:
        raise TypeError(
            "wrap_angle_params expects a mapping of parameter names to "
            "values (e.g. a dict or a pandas DataFrame), but got "
            f"{type(samples).__name__}."
        ) from error

    out = samples.copy()
    if "ecliptic_longitude" in keys:
        out["ecliptic_longitude"] = out["ecliptic_longitude"] % (2 * jnp.pi)
    if "ecliptic_latitude" in keys:
        latitude = out["ecliptic_latitude"]
        out["ecliptic_latitude"] = jnp.pi / 2 - jnp.abs(
            (latitude + jnp.pi / 2) % (2 * jnp.pi) - jnp.pi
        )
    if "inclination" in keys:
        inclination = out["inclination"]
        out["inclination"] = jnp.pi - jnp.abs(inclination % (2 * jnp.pi) - jnp.pi)
    if "polarization" in keys:
        out["polarization"] = out["polarization"] % jnp.pi
    if "reference_phase" in keys:
        out["reference_phase"] = out["reference_phase"] % (2 * jnp.pi)
    return out
