"""
Dagster assets for the Dilbert-o-Matic pipeline.
Each asset represents a step in the MLOps workflow.
"""

# The __all__ variable tells Dagster which assets to load
# when 'load_assets_from_modules' is called in definitions.py

from .drift_detection import jargon_drift_detector
from .weak_labeler import weak_labeler 
from .data_versioner import data_versioner

# --- We will implement these assets next ---
# from .model_training import model_trainer
# from .model_qa import model_qa_gate
# from .model_registry import model_registry_promoter
# from .model_deployment import build_new_bento, retag_production

__all__ = [
    "jargon_drift_detector",
    "weak_labeler",
    "data_versioner",
    # --- Commented out until we implement them ---
    # "model_trainer",
    # "model_qa_gate",
    # "model_registry_promoter",
    # "build_new_bento",
    # "retag_production",
]