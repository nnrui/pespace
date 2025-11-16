import taichi as ti


@ti.kernel
def _compute_whittle_likelihood(
    channels: ti.template(),
    observed_data: ti.template(),
    response_data: ti.template(),
    psd: ti.template(),
    df: float,
) -> float:
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
