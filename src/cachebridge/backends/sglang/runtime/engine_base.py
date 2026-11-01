import asyncio, time
from cachebridge.backends.sglang.runtime import config as _config


def engine_args(role):
    return {
        "model_path": str(
            _config.SOURCE_MODEL if role == "source" else _config.TARGET_MODEL
        ),
        "tp_size": 4,
        "base_gpu_id": 0 if role == "source" else 4,
        "port": 31900 if role == "source" else 31901,
        "mem_fraction_static": 0.55 if role == "source" else 0.90,
        "context_length": 40960,
        "page_size": 1,
        "disable_custom_all_reduce": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "skip_tokenizer_init": False,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "enable_hierarchical_cache": False,
        "max_total_tokens": _config.POOL_TOKENS,
        "max_running_requests": _config.MAX_RUNNING_REQUESTS,
        "chunked_prefill_size": 8192,
        "enable_cache_report": True,
        "log_level": "warning",
    }
