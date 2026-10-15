"""Minimal, opt-in infrastructure for future PGDF major-revision runs.

The package deliberately does not import or alter any historical training
entry point.  A future runner must explicitly construct and call these tools.
"""

from .offline_diagnostics import OracleLabelRecord, diagnose_selection_update
from .profiling import ProfilingMode, RunProfiler
from .selection_logging import SelectionLogger, SelectionSampleRecord

__all__ = [
    "OracleLabelRecord",
    "ProfilingMode",
    "RunProfiler",
    "SelectionLogger",
    "SelectionSampleRecord",
    "diagnose_selection_update",
]
