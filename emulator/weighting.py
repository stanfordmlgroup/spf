import torch


def broadcast_latitude_weights(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast latitude weights to the spatial axis that matches target."""
    lat_len = weights.shape[-1]
    if target.shape[-2] == lat_len:
        view_shape = [weights.shape[0], 1, 1, lat_len, 1]
    elif target.shape[-1] == lat_len:
        view_shape = [weights.shape[0], 1, 1, 1, lat_len]
    else:
        raise ValueError(
            f"Latitude length {lat_len} does not match target height/width {target.shape[-2:]}."
        )
    return weights.view(view_shape)
