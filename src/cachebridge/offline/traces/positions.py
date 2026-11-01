"""Exact Qwen ChatML assistant prediction positions from recorded tokens."""

import re
from .. import io


def prediction_positions(tokenizer, record):
    ids = record["history_ids"] + record["query_ids"]
    h, q = record["H"], record["Q"]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise ValueError("Tokenizer round trip changed the recorded token IDs")
    offsets = encoded["offset_mapping"]
    selected = set()
    for match in re.finditer(re.escape(io.ASSISTANT), text):
        begin = match.end()
        end = text.find("<|im_end|>", begin)
        closed = end >= 0
        if not closed:
            end = len(text)
        content = [i for i, (lo, hi) in enumerate(offsets) if hi > begin and lo < end]
        first = next((i for i, (_, hi) in enumerate(offsets) if hi > begin), len(ids))
        targets = set(content) | {first}
        if closed:
            eos = next(i for i, (lo, _) in enumerate(offsets) if lo == end)
            assert ids[eos] == 151645
            targets.add(eos)
        selected.update(i - 1 - h for i in targets if h <= i - 1 < len(ids))
    if not selected or any(p < 0 or p >= q for p in selected):
        raise ValueError("Missing or invalid assistant prediction positions")
    return dict(
        task=record["task_id"],
        H=h,
        Q=q,
        assistant_prediction_positions=sorted(selected),
    )
