"""VERDICT (verdict-gate): evidence-governed keep/reject decisions for changes to LLM agents.

Research preview. VERDICT compares a candidate revision (prompt, skill, memory, model, tool) with its parent on paired
task-level results, against the parent's own run-to-run variation, and records the decision in a replayable evidence
card. The command line (`verdict-gate --help`) covers calibrate, import, compare, next-batch, replay and card render;
from Python, `next_batch` runs a sequential comparison in batches.
"""

from verdict_gate.core.replay import NextBatch, next_batch

__version__ = "0.3.0.dev0"
__all__ = ["NextBatch", "__version__", "next_batch"]
