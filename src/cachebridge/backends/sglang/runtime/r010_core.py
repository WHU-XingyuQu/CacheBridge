"""Common dense/local mapper execution with direct target-pool block writes."""

import torch


def validate_delivery(schema, delivery, block_tokens):
    if schema not in ("dense", "local") or delivery not in ("full", "support"):
        raise ValueError("Unknown mapper schema or delivery")
    if schema == "dense" and delivery != "full":
        raise ValueError("Dense requires all source heads")
    if block_tokens not in (512, 2048):
        raise ValueError("Frozen common block menu is 512 or 2048")


@torch.inference_mode()
def map_write(mapper, block, pool, indices, *, start, source_head_offset):
    if mapper.method not in ("nv_transfer", "cachebridge"):
        raise ValueError("Unsupported affine mapper schema")
    if mapper.method == "nv_transfer" and source_head_offset != 0:
        raise ValueError("Dense input must start at global head zero")
    n = block.shape[2]
    if not 0 <= start < start + n <= len(indices):
        raise ValueError("Invalid output range")
    device = pool._get_key_buffer(pool.start_layer).device
    mapped = mapper.map_page(
        block,
        position_start=start,
        source_head_offset=source_head_offset,
        output_device=device,
    )
    expected = (2, pool.layer_num, n, *pool._get_key_buffer(pool.start_layer).shape[1:])
    if tuple(mapped.shape) != expected or mapped.dtype != pool.dtype:
        raise ValueError("Mapper output differs from target pool geometry/dtype")
    for i, layer in enumerate(
        range(pool.start_layer, pool.start_layer + pool.layer_num)
    ):
        pool.set_kv_buffer(
            None,
            indices[start : start + n],
            mapped[0, i],
            mapped[1, i],
            layer_id_override=layer,
        )


def map_write_received(
    mapper, received, pool, indices, *, start, source_head_offset, block_tokens
):
    """Whole delivery uses preallocated received H plus blockwise direct installation.

    No list of mapped chunks or full-output concatenate exists. Block delivery
    passes at most block_tokens inputs and performs one identical map/write.
    """
    if block_tokens not in (512, 2048):
        raise ValueError("Unsupported block size")
    for offset in range(0, received.shape[2], block_tokens):
        map_write(
            mapper,
            received[:, :, offset : offset + block_tokens],
            pool,
            indices,
            start=start + offset,
            source_head_offset=source_head_offset,
        )
