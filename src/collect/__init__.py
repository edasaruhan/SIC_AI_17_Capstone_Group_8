"""Turkish Layer A/B collection pipeline (S1-6).

Public surface used by the CLI, the tests and the monitoring report.
"""

from .config import CollectionSettings, ConfigError, RetryPolicy, load_collection_settings
from .plan import (
    DesignConfig,
    build_layer_a_specs,
    build_layer_b_specs,
    load_design,
    summarize_plan,
)
from .providers import (
    FatalProviderError,
    ModelConfig,
    ProviderError,
    ProviderResponse,
    RetryableProviderError,
)
from .rate_limit import TokenBucket
from .records import CallRecord, CallSpec
from .runner import RunSummary, execute_call, pending_specs, run_collection
from .storage import RawLogWriter, ResumeState, raw_log_path, scan_raw_log

__all__ = [
    "CallRecord",
    "CallSpec",
    "CollectionSettings",
    "ConfigError",
    "DesignConfig",
    "FatalProviderError",
    "ModelConfig",
    "ProviderError",
    "ProviderResponse",
    "RawLogWriter",
    "RetryPolicy",
    "RetryableProviderError",
    "ResumeState",
    "RunSummary",
    "TokenBucket",
    "build_layer_a_specs",
    "build_layer_b_specs",
    "execute_call",
    "load_collection_settings",
    "load_design",
    "pending_specs",
    "raw_log_path",
    "run_collection",
    "scan_raw_log",
    "summarize_plan",
]
