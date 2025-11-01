"""
Jargon Sensor - File-based trigger for the MLOps pipeline
Watches new_jargon_candidates.txt and triggers the drift detection workflow when changes occur.
"""

from dagster import sensor, DagsterEventType
from pathlib import Path
import json

from .. import constants

@sensor(name="jargon_candidate_sensor")
def jargon_candidate_sensor(context):
    """
    Detects changes to new_jargon_candidates.txt and yields an event to trigger downstream assets.
    This is a simple implementation - in production, you'd use Dagster's built-in file sensors.
    """
    candidates_file = constants.NEW_JARGON_CANDIDATES_FILE
    
    # Check if file exists
    if not candidates_file.exists():
        context.log.info(f"Candidates file not found at {candidates_file}, creating it...")
        candidates_file.touch()
        return
    
    # Read the file
    with open(candidates_file, 'r') as f:
        content = f.read().strip()
    
    if not content:
        context.log.info("No new jargon candidates detected.")
        return
    
    # Parse candidates (one per line)
    candidates = [line.strip() for line in content.split('\n') if line.strip()]
    
    context.log.info(f"Detected {len(candidates)} new jargon candidates: {candidates}")
    
    # Return context for downstream asset
    from dagster import DynamicOutput
    for candidate in candidates:
        yield DynamicOutput(candidate, mapping_key=candidate)
