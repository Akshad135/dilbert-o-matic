"""
Asset: Weak Labeler
Uses a powerful LLM (via Groq) to generate new training data (input/output pairs)
for any new jargon words detected by the drift_detector.
"""

import os
import json
from groq import Groq
from dagster import asset, AssetExecutionContext

# Import constants from our central constants file
from ..constants import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_WEAK_LABELER_PROMPT,
    TRAINING_DATA_FILE,
)

# This asset depends on the 'jargon_drift_detector' asset.
@asset
def weak_labeler(context: AssetExecutionContext, jargon_drift_detector: dict) -> dict:
    """
    Takes new jargon words and generates training pairs using the Groq API.
    Appends these new pairs to the 'training_data.jsonl' file.
    """
    
    # 1. Check if the upstream asset detected any drift
    if not jargon_drift_detector.get("drift_detected"):
        context.log.info("No new jargon detected. Skipping weak labeling.")
        return {"new_pairs_generated": 0, "status": "skipped"}

    new_jargon_list = jargon_drift_detector.get("new_jargon", [])
    if not new_jargon_list:
        context.log.info("Drift was detected, but new jargon list is empty. Skipping.")
        return {"new_pairs_generated": 0, "status": "skipped"}

    # 2. Check for Groq API Key
    if GROQ_API_KEY == "your_key_here" or not GROQ_API_KEY:
        context.log.error(
            "GROQ_API_KEY is not configured. "
            "Set the GROQ_API_KEY environment variable. Skipping."
        )
        return {"new_pairs_generated": 0, "status": "error_api_key_missing"}

    context.log.info(f"Initializing Groq client with model {GROQ_MODEL}...")
    client = Groq(api_key=GROQ_API_KEY)
    
    new_training_pairs = []
    
    # 3. Iterate over each new jargon word and call the API
    for jargon_word in new_jargon_list:
        context.log.info(f"Generating training pairs for new jargon: '{jargon_word}'")
        
        # Format the prompt from constants.py
        prompt = GROQ_WEAK_LABELER_PROMPT.format(jargon=jargon_word)
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    # System prompt helps enforce JSON-only output
                    {"role": "system", "content": "You are a JSON data generator. Your output MUST be a single, valid JSON object."},
                    {"role": "user", "content": prompt},
                ],
                model=GROQ_MODEL,
                temperature=0.7,
                # THIS IS THE FIX YOU SUGGESTED:
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            context.log.debug(f"Groq API JSON response: {response_content}")

            # 4. Parse the API response (new, more robust logic)
            try:
                data = json.loads(response_content)
                generated_pairs = data.get("pairs", [])
                
                if not generated_pairs:
                    context.log.warning(f"Groq returned valid JSON, but the 'pairs' list was empty or missing for '{jargon_word}'.")
                    continue # Go to the next jargon word

                for pair in generated_pairs:
                    if "input" in pair and "output" in pair:
                        new_training_pairs.append(pair)
                    else:
                        context.log.warning(f"Invalid pair object from Groq: {pair}")
            
            except json.JSONDecodeError:
                context.log.error(f"Failed to decode the *entire* JSON object from Groq for '{jargon_word}': {response_content}")
                continue # Go to the next jargon word

        except Exception as e:
            context.log.error(f"Error calling Groq API for '{jargon_word}': {e}")
            continue

    # 5. Append new data to the main training file
    if new_training_pairs:
        context.log.info(f"Generated {len(new_training_pairs)} new training pairs.")
        try:
            with open(TRAINING_DATA_FILE, 'a') as f:  # 'a' to append
                for pair in new_training_pairs:
                    # Write each pair as a new JSON line
                    f.write(json.dumps(pair) + "\n")
            
            context.log.info(f"Successfully appended {len(new_training_pairs)} pairs to {TRAINING_DATA_FILE}")
            return {
                "new_pairs_generated": len(new_training_pairs),
                "status": "success",
                "new_jargon_processed": new_jargon_list
            }
        except Exception as e:
            context.log.error(f"Failed to write to {TRAINING_DATA_FILE}: {e}")
            return {"new_pairs_generated": 0, "status": "error_file_write"}
    else:
        context.log.info("No new training pairs were successfully generated.")
        return {"new_pairs_generated": 0, "status": "no_pairs_generated"}