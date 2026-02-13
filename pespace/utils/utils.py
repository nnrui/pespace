"""Utility functions and data structures."""
import h5py
import taichi as ti
import taichi.math as tm
import numpy as np
from numpy.typing import NDArray

from .constants import *

PolarizationStruct = ti.types.struct(
    plus=ti.types.matrix(3, 3, float), cross=ti.types.matrix(3, 3, float)
)
ti_complex = ti.types.vector(2, float)
SingleLinkStructComplex = ti.types.struct(
    link12=ti_complex,
    link21=ti_complex,
    link23=ti_complex,
    link32=ti_complex,
    link31=ti_complex,
    link13=ti_complex,
)
SingleLinkStructReal = ti.types.struct(
    link12=float,
    link21=float,
    link23=float,
    link32=float,
    link31=float,
    link13=float,
)


@ti.func
def next_power_of_2(n: ti.u32) -> ti.u32:
    """Compute the next power of 2 greater than or equal to n.

    Parameters
    ----------
    n : ti.u32
        Input unsigned 32-bit integer.

    Returns
    -------
    ti.u32
        The smallest power of 2 that is greater than or equal to n.
    """
    ret = ti.u32(0)
    if n <= ti.u32(1):
        ret = ti.u32(1)
    else:
        n -= ti.u32(1)
        n |= ti.bit_shr(n, ti.u8(1))
        n |= ti.bit_shr(n, ti.u8(2))
        n |= ti.bit_shr(n, ti.u8(4))
        n |= ti.bit_shr(n, ti.u8(8))
        n |= ti.bit_shr(n, ti.u8(16))
        ret = n + ti.u32(1)
    return ret


@ti.func
def sinc(x: float) -> float:
    """Compute the sinc function: sin(x)/x.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        The value of sinc(x). Returns 1.0 when x = 0.0.
    """
    ret = 0.0
    if x == 0.0:
        ret = 1.0
    else:
        ret = tm.sin(x) / x
    return ret


@ti.func
def linear_interpolate_kernel(left, right, frac) -> float:
    """Perform linear interpolation between two values.

    Parameters
    ----------
    left : float
        Left boundary value.
    right : float
        Right boundary value.
    frac : float
        Fractional position between left and right, in the range [0, 1].

    Returns
    -------
    float
        Interpolated value.
    """
    return left + (right - left) * frac


@ti.func
def lagrange_interpolate_kernel():
    """Lagrange interpolation kernel (not implemented).

    Notes
    -----
    This function is a placeholder for future implementation.
    """
    pass


@ti.func
def sinc_interpolate_kernel():
    """Sinc interpolation kernel (not implemented).

    Notes
    -----
    This function is a placeholder for future implementation.
    """
    pass


INTERPOLATE_KERNELS = {
    "linear": linear_interpolate_kernel,
    "sinc": sinc_interpolate_kernel,
    "lagrange": lagrange_interpolate_kernel,
}


@ti.func
def get_polarization_tensor_ssb(
    lam: float, beta: float, psi: float
) -> PolarizationStruct:
    """Compute the polarization tensors in the Solar System Barycenter frame.

    Parameters
    ----------
    lam : float
        Ecliptic longitude in radians.
    beta : float
        Ecliptic latitude in radians, in the range :math:`(-\\pi/2, \\pi/2)`.
    psi : float
        Polarization angle in radians.

    Returns
    -------
    PolarizationStruct
        A struct containing two 3x3 matrices:
        
        - plus: Plus polarization tensor (:math:`e_+ = p \\otimes p - q \\otimes q`)
        - cross: Cross polarization tensor (:math:`e_\\times = p \\otimes q + q \\otimes p`)
    """
    # TODO: the constant should compute only once to reduce compution burden.
    sin_lam = tm.sin(lam)
    cos_lam = tm.cos(lam)
    sin_beta = tm.sin(beta)
    cos_beta = tm.cos(beta)
    sin_psi = tm.sin(psi)
    cos_psi = tm.cos(psi)
    p = ti.Vector(
        [
            sin_lam * cos_psi - cos_lam * sin_beta * sin_psi,
            -cos_lam * cos_psi - sin_lam * sin_beta * sin_psi,
            cos_beta * sin_psi,
        ]
    )
    q = ti.Vector(
        [
            -sin_lam * sin_psi - cos_lam * sin_beta * cos_psi,
            cos_lam * sin_psi - sin_lam * sin_beta * cos_psi,
            cos_beta * cos_psi,
        ]
    )

    return PolarizationStruct(
        plus=(p.outer_product(p) - q.outer_product(q)),
        cross=(p.outer_product(q) + q.outer_product(p)),
    )


@ti.func
def get_gw_propagation_unit_vector(
    lam: float, beta: float
) -> ti.types.vector(3, float):
    """Compute the gravitational wave propagation unit vector.

    Parameters
    ----------
    lam : float
        Ecliptic longitude in radians.
    beta : float
        Ecliptic latitude in radians, in the range :math:`(-\\pi/2, \\pi/2)`.

    Returns
    -------
    ti.types.vector(3, float)
        Unit vector pointing in the direction of GW propagation in the SSB frame.

    Notes
    -----
    The propagation direction is opposite to the source direction.
    """
    return ti.Vector(
        [-tm.cos(beta) * tm.cos(lam), -tm.cos(beta) * tm.sin(lam), -tm.sin(beta)]
    )


def taichi_field_to_complex_numpy_array_dict(
    field_container: ti.Field,
) -> dict[str, NDArray]:
    """Convert a Taichi field to a dictionary of complex NumPy arrays.

    Parameters
    ----------
    field_container : ti.Field
        Taichi field container with shape (N, 2), where the last dimension
        represents [real, imaginary] components.

    Returns
    -------
    dict[str, NDArray]
        Dictionary mapping field names to complex-valued NumPy arrays of shape (N,).
    """
    return dict(
        [
            (key, data[:, 0] + 1j * data[:, 1])
            for key, data in field_container.to_numpy().items()
        ]
    )


def complex_numpy_array_dict_to_taichi_field(
    array_dict: dict[str, NDArray],
    field_container: ti.Field,
) -> None:
    """Convert a dictionary of complex NumPy arrays to a Taichi field.

    Parameters
    ----------
    array_dict : dict[str, NDArray]
        Dictionary mapping field names to 1D complex-valued NumPy arrays.
    field_container : ti.Field
        Target Taichi field container with shape (N, 2) to store
        [real, imaginary] components.
    """
    field_container.from_numpy(
        dict(
            [
                (key, np.column_stack([data.real, data.imag]))
                for key, data in array_dict.items()
            ]
        )
    )


# def cutoff_frequency_PhenomD(mass_1, mass_2):
#     '''
#     return the high frequency cutoff in Hz, using Mf=0.2 copied form LALSimIMRPhenomD.h,
#     which could be used in determining the sampling frequency in TD or the frequency bound in FD for SMBH.

#     Parameters
#     ==========
#     mass_1: mass of heavier object in Msun
#     mass_2: mass of lighter object in Msun

#     Returns:
#     ========
#     f_cut: in Hz
#     '''
#     total_mass = mass_1 + mass_2
#     M_sec = total_mass * MTSUN_SI
#     f_cut = Mf_CUT_PhenomD/M_sec
#     return f_cut


# def start_frequency():
#     '''
#     description

#     Parameters
#     ==========


#     Returns:
#     ========

#     '''
#     return


# def time_in_band_leading_order(mass_1, mass_2, start_frequency, safety_factor=1.1):
#     '''
#     TODO consider the noise not only the start_frequency
#     time to merger from the minimum_frequency
#     note that the minimum_frequency maybe higher than the low frequency cutoff of the detector
#     the returned time is a rough approximation with the lead oder

#     Parameters
#     ==========
#     mass_1: mass of heavier object in Msun
#     mass_2: mass of lighter object in Msun
#     start_frequency: in Hz
#     safety_factor: multiplicitive safety factor

#     Returns:
#     ========
#     time_length: in second
#     '''
#     total_mass = mass_1 + mass_2
#     M_sec = total_mass * MTSUN_SI
#     Mf_start = M_sec * start_frequency
#     eta = component_masses_to_symmetric_mass_ratio(mass_1, mass_2)
#     # dimensionless unit
#     time_to_merger = 5/256 / eta * (PI*Mf_start)**(-8/3)
#     # convert to unit of second
#     time_to_merger *= M_sec
#     time_length = time_to_merger*safety_factor
#     return time_length


# def estimate_imr_duration(mass_1, mass_2, chi_1, chi_2, start_frequency, safety_factor=1.1):
#     '''
#     deprecate, do not use this func, have unknown error, return an negtive value for SMBH.
#     '''
#     time_length = lalsim.SimIMRPhenomDChirpTime(mass_1*MSUN_SI, mass_2*MSUN_SI, chi_1, chi_2, start_frequency)
#     time_length *= safety_factor
#     return time_length


# def post_merger_time_SMBH():
#     '''
#     description

#     Parameters
#     ==========


#     Returns:
#     ========

#     '''
#     return

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """Recursively save a dictionary to an HDF5 group.

    Parameters
    ----------
    h5file : h5py.File
        HDF5 file object.
    path : str
        Path inside the HDF5 file where the dictionary will be saved.
    dic : dict
        Dictionary containing the data to save. Supports nested dictionaries,
        lists, NumPy arrays, and None values.

    Raises
    ------
    ValueError
        If the dictionary contains values of unsupported types.

    Notes
    -----
    This function is adapted from ``bilby.core.utils.io.recursively_save_dict_contents_to_group``.
    Supported value types: ``dict``, ``list``, ``np.ndarray``, ``None``.
    """
    for key, value in dic.items():
        if isinstance(value, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + "/", value)
        elif isinstance(value, list):
            if len(value) == 0:
                h5file[path + key] = h5py.Empty("f")
            else:
                for idx, item in enumerate(value):
                    recursively_save_dict_contents_to_group(
                        h5file, path + key + "/" + f"item_{idx}" + "/", item
                    )
        elif isinstance(value, np.ndarray):
            h5file[path + key] = value
        elif value is None:
            h5file[path + key] = h5py.Empty("f")
        else:
            raise ValueError(f"Cannot save {key}: {type(value)} type")


def recursively_load_dict_contents_from_group(h5file, path):
    """Recursively load an HDF5 group into a dictionary.

    Parameters
    ----------
    h5file : h5py.File
        HDF5 file object.
    path : str
        Path within the HDF5 file to load from.

    Returns
    -------
    dict
        Dictionary containing the contents of the HDF5 group, with nested
        structure preserved.

    Notes
    -----
    This function is adapted from ``bilby.core.utils.io.recursively_load_dict_contents_from_group``.
    """
    output = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            output[key] = item[()]
        elif isinstance(item, h5py.Group):
            output[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return output
