"""Head-local engineering baseline: select supported heads before preprocessing."""

from ..mapper import PageMapper as FullInputMapper


class PageMapper(FullInputMapper):
    def map_page(
        self,
        source_page,
        *,
        position_start,
        source_head_offset=0,
        output_device="cpu",
        output=None,
    ):
        if self.method == "cachebridge":
            start = self.target_head_start - source_head_offset
            stop = self.target_head_stop - source_head_offset
            if not 0 <= start < stop <= source_page.shape[3]:
                raise ValueError(
                    "Received source shard does not contain required heads"
                )
            # A view into received storage: no fewer bytes delivered, no custom
            # kernel and no changed head-local projection/weights/precision.
            source_page = source_page[:, :, :, start:stop, :]
            source_head_offset = self.target_head_start
        return super().map_page(
            source_page,
            position_start=position_start,
            source_head_offset=source_head_offset,
            output_device=output_device,
            output=output,
        )
