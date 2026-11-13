"""Adapters: turn recorded run artifacts into TaskOutcome lists plus a RunManifest.

Adapters never compute decisions. A field the artifacts do not record is never guessed: strings become "UNKNOWN",
integers -1, floats NaN, a seed None and a timestamp 1970-01-01T00:00Z, and the field is listed in
AdaptedRun.unknown_fields.
"""

from verdict_gate.adapters.common import UNKNOWN, UNKNOWN_FLOAT, UNKNOWN_INT, UNKNOWN_TIME, AdaptedRun

__all__ = ["UNKNOWN", "UNKNOWN_FLOAT", "UNKNOWN_INT", "UNKNOWN_TIME", "AdaptedRun"]
