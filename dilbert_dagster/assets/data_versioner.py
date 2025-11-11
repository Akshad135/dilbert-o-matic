"""
Asset: Data Versioner
Versions all data assets (training data, known jargon) using DVC
and cleans up the candidate file to "close the loop".
"""

import subprocess
import os
import pandas as pd
from dagster import asset, AssetExecutionContext

# Import constants from our central constants file
from ..constants import (
    PROJECT_ROOT,
    TRAINING_DATA_FILE,
    KNOWN_JARGON_FILE,
    NEW_JARGON_CANDIDATES_FILE
)

# This asset now depends on *both* upstream assets
@asset
def data_versioner(
    context: AssetExecutionContext, 
    weak_labeler: dict, 
    jargon_drift_detector: dict
) -> dict:
    """
    If weak_labeler was successful:
    1. Appends new jargon to known_jargon.csv
    2. Runs 'dvc add' on both training_data.jsonl and known_jargon.csv
    3. Runs 'dvc push'
    4. Clears new_jargon_candidates.txt
    """
    
    # 1. Check upstream status
    weak_labeler_status = weak_labeler.get("status")
    if weak_labeler_status != "success":
        context.log.info(f"Weak labeler status was '{weak_labeler_status}'. No new data to version. Skipping.")
        return {"status": "skipped_no_new_data"}
        
    new_pairs = weak_labeler.get("new_pairs_generated", 0)
    if new_pairs == 0:
        context.log.info("Weak labeler ran but generated 0 pairs. No new data to version. Skipping.")
        return {"status": "skipped_no_new_data"}
        
    context.log.info(f"Weak labeler added {new_pairs} new pairs. Versioning all data assets...")

    # 2. Get the new jargon list from the drift detector
    new_jargon_list = jargon_drift_detector.get("new_jargon", [])

    if not new_jargon_list:
        context.log.warning("Weak labeler succeeded, but new jargon list was empty. Skipping data versioning.")
        return {"status": "skipped_no_new_jargon_list"}

    # 3. Append new jargon to known_jargon.csv
    try:
        new_jargon_df = pd.DataFrame(new_jargon_list, columns=['jargon'])
        # Append to the CSV, don't write header if file already exists
        new_jargon_df.to_csv(
            KNOWN_JARGON_FILE, 
            mode='a', 
            header=not KNOWN_JARGON_FILE.exists(), 
            index=False
        )
        context.log.info(f"Appended {len(new_jargon_list)} words to {KNOWN_JARGON_FILE.name}")
    except Exception as e:
        context.log.error(f"Failed to append new jargon to {KNOWN_JARGON_FILE.name}: {e}")
        return {"status": "error_writing_known_jargon"}

    # 4. Helper function to run shell commands (no changes)
    def run_dvc_command(cmd: list):
        context.log.info(f"Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            context.log.info(f"DVC STDOUT: {result.stdout}")
            if result.stderr:
                context.log.warning(f"DVC STDERR: {result.stderr}")
            return True
        except subprocess.CalledProcessError as e:
            context.log.error(f"DVC command failed: {e.stderr}")
            return False
        except FileNotFoundError:
            context.log.error(f"DVC command not found. Is DVC installed and in the PATH?")
            return False

    # 5. Get relative paths for DVC
    try:
        rel_training_data_path = os.path.relpath(TRAINING_DATA_FILE, PROJECT_ROOT)
        rel_jargon_file_path = os.path.relpath(KNOWN_JARGON_FILE, PROJECT_ROOT)
    except Exception as e:
        context.log.error(f"Could not calculate relative paths: {e}. Aborting.")
        return {"status": "error_path_calculation"}

    # 6. Run 'dvc add' on BOTH files
    if not run_dvc_command(["dvc", "add", rel_training_data_path]):
        return {"status": "error_dvc_add_training_data"}
    
    if not run_dvc_command(["dvc", "add", rel_jargon_file_path]):
        return {"status": "error_dvc_add_known_jargon"}
        
    # 7. Run 'dvc push'
    if not run_dvc_command(["dvc", "push"]):
        return {"status": "error_dvc_push"}
        
    context.log.info(f"Successfully versioned and pushed {rel_training_data_path} and {rel_jargon_file_path} to DVC remote.")
    
    # 8. Clear the candidates file to "close the loop"
    try:
        with open(NEW_JARGON_CANDIDATES_FILE, 'w') as f:
            f.write("") # Truncate the file
        context.log.info(f"Cleared {NEW_JARGON_CANDIDATES_FILE.name} for next run.")
    except Exception as e:
        context.log.error(f"Failed to clear {NEW_JARGON_CANDIDATES_FILE.name}: {e}")
        return {"status": "error_clearing_candidates"}

    return {
        "status": "success",
        "versioned_files": [rel_training_data_path, rel_jargon_file_path]
    }