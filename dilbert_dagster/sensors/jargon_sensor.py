"""
Dagster Sensor - Jargon Candidate Detector
Watches new_jargon_candidates.txt and triggers the data processing job.
"""

import os
from dagster import sensor, RunRequest, SkipReason, SensorEvaluationContext

# Import the constant from our central file
from ..constants import NEW_JARGON_CANDIDATES_FILE

def _get_mtime(file_path):
    """Get the modification time of a file."""
    return os.path.getmtime(file_path)

@sensor(job_name="process_new_jargon_job")
def jargon_candidate_sensor(context: SensorEvaluationContext):
    """
    Checks for modifications to the candidates file and triggers 
    the 'process_new_jargon_job' if changes are found.
    """
    
    # Use the imported constant directly
    monitor_file = NEW_JARGON_CANDIDATES_FILE
    
    if not monitor_file.exists():
        return SkipReason(f"File {monitor_file.name} not found.")

    last_mtime = float(context.cursor or 0)
    current_mtime = _get_mtime(monitor_file)

    if current_mtime > last_mtime:
        context.log.info(f"New data detected in {monitor_file.name}. Triggering job.")
        context.update_cursor(str(current_mtime))
        # Yield a RunRequest for the job named "process_new_jargon_job"
        return RunRequest(run_key=f"jargon_sensor_{current_mtime}")
    
    return SkipReason(f"No new data in {monitor_file.name}.")