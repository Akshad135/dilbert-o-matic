"""
Asset: Data Versioner
Versions the training data using DVC after it's updated by the weak_labeler.
This creates an immutable, versioned snapshot of the data used for a training run.
"""

import subprocess
import os
from dagster import asset, AssetExecutionContext

# Import constants from our central constants file
from ..constants import (
    PROJECT_ROOT,
    TRAINING_DATA_FILE
)

# This asset depends on the 'weak_labeler'
# It will only run after 'weak_labeler' successfully completes.
@asset
def data_versioner(context: AssetExecutionContext, weak_labeler: dict) -> dict:
    """
    Runs 'dvc add' and 'dvc push' on the training data file 
    if the weak_labeler added new pairs.
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
        
    context.log.info(f"Weak labeler added {new_pairs} new pairs. Versioning {TRAINING_DATA_FILE.name} with DVC...")

    # 2. Define helper function to run shell commands
    def run_dvc_command(cmd: list):
        """Helper to run a subprocess command from the project root."""
        context.log.info(f"Running command: {' '.join(cmd)}")
        try:
            # Run command from the project root, using the constant
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
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

    # 3. Get the relative path for the training file (DVC works better with relative paths)
    # We use os.path.relpath to ensure it's correct
    try:
        relative_data_path = os.path.relpath(TRAINING_DATA_FILE, PROJECT_ROOT)
    except Exception as e:
        context.log.error(f"Could not calculate relative path: {e}. Using filename.")
        # Fallback to just the filename
        relative_data_path = TRAINING_DATA_FILE.name

    # 4. Run 'dvc add'
    # We are adding the file specified in constants.py
    add_cmd = ["dvc", "add", relative_data_path]
    if not run_dvc_command(add_cmd):
        context.log.error("Failed to 'dvc add' the data file.")
        return {"status": "error_dvc_add"}
        
    # 5. Run 'dvc push'
    push_cmd = ["dvc", "push"]
    if not run_dvc_command(push_cmd):
        context.log.error("Failed to 'dvc push' the data file.")
        return {"status": "error_dvc_push"}
        
    context.log.info(f"Successfully versioned and pushed {relative_data_path} to DVC remote.")
    
    return {
        "status": "success",
        "versioned_file": relative_data_path
    }