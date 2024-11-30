# TODO:
# - current implementation is awkward, improvements in modulariztion are required;
# - add more type combination, and well document in their features like sensitive to specific polarization;
import taichi as ti
import taichi.math as tm

from ..utils import ComplexNumber


SingleLinksStruct = ti.types.struct(
    link12=ComplexNumber,
    link21=ComplexNumber,
    link23=ComplexNumber,
    link32=ComplexNumber,
    link31=ComplexNumber,
    link13=ComplexNumber,
)


@ti.func
def _TDI_X_FD(
    z: ComplexNumber, singlelink_responses: SingleLinksStruct
) -> ComplexNumber:
    """
    Function for computing X channel of TDI combination in frequency domain.

    Parameters:
    ===========
    z:
        Delay factor, exp(-1j*2*PI*f*arm_length_sec).
    singlelink_responses:
        Responses of each link.

    Returns:
    ========
    X channel without the generation prefactor.
    """
    return (
        singlelink_responses["link31"]
        + tm.cmul(z, singlelink_responses["link13"])
        - singlelink_responses["link21"]
        - tm.cmul(z, singlelink_responses["link12"])
    )


@ti.func
def _TDI_Y_FD(
    z: ComplexNumber, singlelink_responses: SingleLinksStruct
) -> ComplexNumber:
    """
    Function for computing Y channel of TDI combination in frequency domain.

    Parameters:
    ===========
    z:
        Delay factor, exp(-1j*2*PI*f*arm_length_sec).
    singlelink_responses:
        Responses of each link.

    Returns:
    ========
    Y channel without the generation prefactor.
    """
    return (
        singlelink_responses["link12"]
        + tm.cmul(z, singlelink_responses["link21"])
        - singlelink_responses["link32"]
        - tm.cmul(z, singlelink_responses["link23"])
    )


@ti.func
def _TDI_Z_FD(
    z: ComplexNumber, singlelink_responses: SingleLinksStruct
) -> ComplexNumber:
    """
    Function for computing Z channel of TDI combination in frequency domain.

    Parameters:
    ===========
    z:
        Delay factor, exp(-1j*2*PI*f*arm_length_sec).
    singlelink_responses:
        Responses of each link.

    Returns:
    ========
    Z channel without the generation prefactor.
    """
    return (
        singlelink_responses["link23"]
        + tm.cmul(z, singlelink_responses["link32"])
        - singlelink_responses["link13"]
        - tm.cmul(z, singlelink_responses["link31"])
    )


@ti.func
def _TDI_A_FD(
    z: ComplexNumber, singlelink_responses: SingleLinksStruct
) -> ComplexNumber:
    """
    Function for computing A channel of TDI combination in frequency domain.

    Parameters:
    ===========
    z:
        Delay factor, exp(-1j*2*PI*f*arm_length_sec).
    singlelink_responses:
        Responses of each link.

    Returns:
    ========
    A channel without the generation prefactor.
    """
    return (
        singlelink_responses["link23"]
        + tm.cmul(z, singlelink_responses["link32"])
        + singlelink_responses["link21"]
        + tm.cmul(z, singlelink_responses["link12"])
        - tm.cmul(
            (ComplexNumber(1, 0) + z),
            (singlelink_responses["link13"]) + singlelink_responses["link31"],
        )
    ) / tm.sqrt(2)


@ti.func
def _TDI_E_FD(
    z: ComplexNumber, singlelink_responses: SingleLinksStruct
) -> ComplexNumber:
    """
    Function for computing E channel of TDI combination in frequency domain.

    Parameters:
    ===========
    z:
        Delay factor, exp(-1j*2*PI*f*arm_length_sec).
    singlelink_responses:
        Responses of each link.

    Returns:
    ========
    E channel without the generation prefactor.
    """
    return (
        tm.cmul(
            (ComplexNumber(1, 0) - z),
            (singlelink_responses["link31"] - singlelink_responses["link13"]),
        )
        + tm.cmul(
            (z + ComplexNumber(2, 0)),
            (singlelink_responses["link32"] - singlelink_responses["link12"]),
        )
        + tm.cmul(
            (ComplexNumber(1, 0) + 2 * z),
            (singlelink_responses["link23"] - singlelink_responses["link21"]),
        )
    ) / tm.sqrt(6)


@ti.func
def _TDI_T_FD(
    z: ComplexNumber, singlelink_responses: SingleLinksStruct
) -> ComplexNumber:
    """
    Function for computing T channel of TDI combination in frequency domain.

    Parameters:
    ===========
    z:
        Delay factor, exp(-1j*2*PI*f*arm_length_sec).
    singlelink_responses:
        Responses of each link.

    Returns:
    ========
    T channel without the generation prefactor.
    """
    return (
        tm.cmul(
            (
                singlelink_responses["link12"]
                - singlelink_responses["link21"]
                + singlelink_responses["link23"]
                - singlelink_responses["link32"]
                + singlelink_responses["link31"]
                - singlelink_responses["link13"]
            ),
            (ComplexNumber(1, 0) - z),
        )
    ) / tm.sqrt(3)


TDI_combine_function_FD = {
    "X": _TDI_X_FD,
    "Y": _TDI_Y_FD,
    "Z": _TDI_Z_FD,
    "A": _TDI_A_FD,
    "E": _TDI_E_FD,
    "T": _TDI_T_FD,
}
implemented_TDI_generations = ("1.5", "2.0")
implemented_TDI_channels = ("X", "Y", "Z", "A", "E", "T")
