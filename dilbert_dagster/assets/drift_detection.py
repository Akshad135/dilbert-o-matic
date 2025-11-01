"""
Asset: Jargon Drift Detector
Checks if new jargon words are not in the known jargon list.
"""
import pandas as pd
from dagster import asset, context
from dilbert_dagster.constants import KNOWN_JARGON_FILE, NEW_JARGON_CANDIDATES_FILE


@asset
def jargon_drift_detector(context) -> dict:
    """
    Detects if new jargon words represent a data drift.
    Returns metadata about detected drift.
    """
    try:
        # Load known jargon
        known_jargon_df = pd.read_csv(KNOWN_JARGON_FILE)
        known_jargon = set(known_jargon_df['jargon'].str.lower())
        
        # Load new candidates
        with open(NEW_JARGON_CANDIDATES_FILE, 'r') as f:
            new_candidates = [line.strip().lower() for line in f if line.strip()]
        
        # Detect drift
        new_jargon = [word for word in new_candidates if word not in known_jargon]
        
        drift_detected = len(new_jargon) > 0
        
        context.log.info(f"Drift detection: {len(new_jargon)} new jargon words detected")
        context.log.info(f"New jargon: {new_jargon}")
        
        return {
            "drift_detected": drift_detected,
            "new_jargon": new_jargon,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
    
    except FileNotFoundError as e:
        context.log.warning(f"File not found: {e}. Initializing empty files.")
        return {"drift_detected": False, "new_jargon": [], "timestamp": pd.Timestamp.now().isoformat()}
