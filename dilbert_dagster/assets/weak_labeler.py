"""
Uses Groq to generate new training pairs for detected jargon.
"""
import os
import json
from groq import Groq
from dagster import asset, AssetExecutionContext
from ..constants import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_WEAK_LABELER_PROMPT,
    TRAINING_DATA_FILE,
)

@asset
def weak_labeler(context: AssetExecutionContext, jargon_drift_detector: dict) -> dict:
    """Generates and appends new training pairs based on drift report."""
    
    if not jargon_drift_detector.get("drift_detected"):
        context.log.info("No new jargon detected. Skipping weak labeling.")
        return {"new_pairs_generated": 0, "status": "skipped"}

    new_jargon_list = jargon_drift_detector.get("new_jargon", [])
    if not new_jargon_list:
        context.log.info("Drift was detected, but new jargon list is empty. Skipping.")
        return {"new_pairs_generated": 0, "status": "skipped"}

    if GROQ_API_KEY == "your_key_here" or not GROQ_API_KEY:
        context.log.error("GROQ_API_KEY is not configured. Skipping.")
        return {"new_pairs_generated": 0, "status": "error_api_key_missing"}

    context.log.info(f"Initializing Groq client with model {GROQ_MODEL}...")
    client = Groq(api_key=GROQ_API_KEY)
    
    new_training_pairs = []
    
    for jargon_word in new_jargon_list:
        context.log.info(f"Generating training pairs for new jargon: '{jargon_word}'")
        prompt = GROQ_WEAK_LABELER_PROMPT.format(jargon=jargon_word)
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a JSON data generator. Your output MUST be a single, valid JSON object."},
                    {"role": "user", "content": prompt},
                ],
                model=GROQ_MODEL,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            response_content = chat_completion.choices[0].message.content
            context.log.debug(f"Groq API JSON response: {response_content}")

            try:
                data = json.loads(response_content)
                generated_pairs = data.get("pairs", [])
                
                if not generated_pairs:
                    context.log.warning(f"Groq returned valid JSON, but 'pairs' list was empty for '{jargon_word}'.")
                    continue 

                for pair in generated_pairs:
                    if "input" in pair and "output" in pair:
                        new_training_pairs.append(pair)
                    else:
                        context.log.warning(f"Invalid pair object from Groq: {pair}")
            
            except json.JSONDecodeError:
                context.log.error(f"Failed to decode JSON object from Groq for '{jargon_word}': {response_content}")
                continue

        except Exception as e:
            context.log.error(f"Error calling Groq API for '{jargon_word}': {e}")
            continue

    if new_training_pairs:
        context.log.info(f"Generated {len(new_training_pairs)} new training pairs.")
        try:
            with open(TRAINING_DATA_FILE, 'a') as f:
                for pair in new_training_pairs:
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