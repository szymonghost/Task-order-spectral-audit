"""Reusable spectral audits for complete task-order landscapes.

The public API deliberately separates three concerns:

``CompleteLandscape``
    One or more validated complete functions on :math:`S_K`; repeated columns
    may serve as independent seed landscapes for inference.
``analyze_landscape``
    Exact isotypic decomposition, optional seed-aware signal inference, and
    requested dimension-normalized within-order allocation contrasts.
``publish_audit``
    Atomic CSV publication and a manifest binding every output to its input.

No experiment paths, task batteries, or NPZ field names are hard-coded.
"""

__version__ = "0.2.0"

from .audit import AuditConfig, AuditResult, analyze_landscape, publish_audit
from .allocation import describe_allocation, infer_allocation
from .schema import CompleteLandscape, load_npz_landscape

__all__ = [
    "AuditConfig",
    "AuditResult",
    "CompleteLandscape",
    "analyze_landscape",
    "describe_allocation",
    "infer_allocation",
    "load_npz_landscape",
    "publish_audit",
]
