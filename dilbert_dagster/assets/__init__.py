"""
Dagster assets for the Dilbert-o-Matic pipeline.
Each asset represents a step in the MLOps workflow.
"""

from .drift_detection import jargon_drift_detector
from .weak_labeling import weak_labeler
from .data_versioning import data_versioner
from .model_training import model_trainer
from .model_qa import model_qa_gate
from .model_registry import model_registry_promoter
from .model_deployment import build_new_bento, retag_production

__all__ = [
    "jargon_drift_detector",
    "weak_labeler",
    "data_versioner",
    "model_trainer",
    "model_qa_gate",
    "model_registry_promoter",
    "build_new_bento",
    "retag_production",
]
