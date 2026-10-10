import math

import torch


def deterministic_even_indices(length, count, device=None):
    """Return `count` evenly spaced int64 indices in [0, length - 1]."""
    length = int(length)
    count = int(count)
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if count <= 0 or count > length:
        raise ValueError(f"count must be in [1, {length}], got {count}")
    if count == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)

    # Integer arithmetic avoids float32 endpoint rounding for activations with
    # more than 2**24 elements. The final index is exactly length - 1.
    steps = torch.arange(count, dtype=torch.int64, device=device)
    return torch.div(
        steps * (length - 1),
        count - 1,
        rounding_mode="floor",
    )


def largest_power_of_two_divisor(channels, minimum_block_size=8):
    """Return the largest power-of-two block that exactly divides channels."""
    if channels < minimum_block_size:
        return 0
    block_size = 1
    while channels % 2 == 0:
        block_size *= 2
        channels //= 2
    return block_size if block_size >= minimum_block_size else 0


def normalized_block_fwht(x, block_size, dim=-1):
    """Apply an orthonormal Walsh-Hadamard transform in independent blocks."""
    if block_size <= 1:
        return x
    if block_size & (block_size - 1):
        raise ValueError(f"Hadamard block size must be a power of two, got {block_size}")

    dim = dim if dim >= 0 else x.dim() + dim
    channels = x.shape[dim]
    if channels % block_size != 0:
        raise ValueError(
            f"Hadamard block size {block_size} does not divide channel dimension {channels}"
        )

    moved = x.movedim(dim, -1)
    original_shape = moved.shape
    y = moved.reshape(*original_shape[:-1], channels // block_size, block_size)
    block_prefix = y.shape[:-1]
    h = 1
    while h < block_size:
        butterflies = y.reshape(*block_prefix, block_size // (2 * h), 2, h)
        left = butterflies[..., 0, :]
        right = butterflies[..., 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(
            *block_prefix, block_size
        )
        h *= 2
    y = y.reshape(original_shape) / math.sqrt(block_size)
    return y.movedim(-1, dim)


def flatten_channel_vectors(x, channel_dim):
    """Convert a Linear/Conv activation tensor to a 2-D [vectors, channels] tensor."""
    channel_dim = channel_dim if channel_dim >= 0 else x.dim() + channel_dim
    return x.movedim(channel_dim, -1).reshape(-1, x.shape[channel_dim])


def rotate_weight_input_axis(weight, block_size):
    """Rotate the input-channel axis of Linear/Conv1d/Conv2d weights."""
    return normalized_block_fwht(weight, block_size, dim=1)
