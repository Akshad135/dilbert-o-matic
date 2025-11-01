"""
Dagster Definitions - Entry Point
This file is automatically discovered by Dagster and defines all assets, jobs, and sensors.
"""

from dagster import Definitions, load_assets_from_modules
from pathlib import Path

from . import constants

# ============================================================================
# IMPORT ASSET MODULES (we'll create these next)
# ============================================================================
# Uncomment these as we create each module:
# from .assets import drift_detector, weak_labeler, data_versioner
from .sensors import jargon_candidate_sensor

# ============================================================================
# LOAD ALL ASSETS
# ============================================================================
# For now, this is empty. We'll populate it as we create assets
all_assets = []

# When you have assets, load them like this:
# all_assets = load_assets_from_modules([drift_detector, weak_labeler, ...])

# ============================================================================
# DEFINE SENSORS
# ============================================================================
# We'll add sensors here as we create them
all_sensors = [jargon_candidate_sensor]

# ============================================================================
# DEFINE JOBS
# ============================================================================
# Jobs wire assets together. We'll add these later.
all_jobs = []

# ============================================================================
# DAGSTER DEFINITIONS
# ============================================================================
defs = Definitions(
    assets=all_assets,
    sensors=all_sensors,
    jobs=all_jobs,
)
