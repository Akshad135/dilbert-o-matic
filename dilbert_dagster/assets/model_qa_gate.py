"""
This asset acts as a "gate" after training. It loads the newly trained
model and a "golden set" of QA examples. It then uses a
SentenceTransformer (embedding) model to compare the semantic similarity
of the model's generated answers against the "golden" answers.

If the average similarity score is above a threshold, the gate "passes".
This result is logged to MLflow in the *same run* as the training.
"""


import torch
import mlflow
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from dagster import asset, AssetExecutionContext
from ..constants import (
    MODEL_V1_DIR,
    QA_GOLDEN_SET_FILE,
    QA_EMBEDDING_MODEL,
    QA_MIN_COSINE_DISTANCE,
    MODEL_MAX_LENGTH,
    INFERENCE_NUM_BEAMS,
    MLFLOW_TRACKING_URI,
)
from .model_trainer import model_trainer


@asset(deps=[model_trainer])
def model_qa_gate(context: AssetExecutionContext, model_trainer: dict) -> dict:
    """Run semantic QA on the newly trained model and log results to the same MLflow run."""
    if model_trainer.get("status") != "success":
        context.log.info("Upstream training failed or was skipped. Skipping QA.")
        return {"status": "skipped", "qa_passed": False}

    run_id = model_trainer.get("run_id")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_id=run_id):
        try:
            context.log.info(f"Loading golden QA set: {QA_GOLDEN_SET_FILE}")
            qa_df = load_dataset("json", data_files=str(QA_GOLDEN_SET_FILE), split="train").to_pandas()
            context.log.info(f"Loaded {len(qa_df)} examples.")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            context.log.info(f"Loading trained model from {MODEL_V1_DIR}")
            trained_model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_V1_DIR)).to(device)
            trained_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_V1_DIR))

            context.log.info(f"Loading embedding model: {QA_EMBEDDING_MODEL}")
            embedding_model = SentenceTransformer(QA_EMBEDDING_MODEL, device=device)

            # Inference on QA inputs
            prefix = "Translate to corporate jargon: "
            prompts = [prefix + s for s in qa_df["input"]]
            batch = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(device)

            context.log.info("Running batch inference...")
            with torch.inference_mode():
                outputs = trained_model.generate(
                    **batch,
                    max_new_tokens=MODEL_MAX_LENGTH,
                    num_beams=INFERENCE_NUM_BEAMS,
                    early_stopping=True,
                )

            generated_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            expected_outputs = qa_df["expected_output"].tolist()

            # Cosine similarity between generated and expected outputs
            expected_embeddings = embedding_model.encode(expected_outputs, convert_to_tensor=True)
            generated_embeddings = embedding_model.encode(generated_outputs, convert_to_tensor=True)
            cosine_scores = util.cos_sim(expected_embeddings, generated_embeddings)
            average_score = torch.diag(cosine_scores).mean().item()

            passed = average_score >= QA_MIN_COSINE_DISTANCE
            context.log.info(f"QA Score: {average_score:.4f} / Threshold: {QA_MIN_COSINE_DISTANCE}")
            context.log.info(f"QA Gate Passed: {passed}")

            mlflow.log_metric("qa_cosine_similarity", average_score)
            mlflow.log_param("qa_gate_passed", str(passed))

            return {"status": "success", "qa_passed": passed, "qa_score": average_score}

        except Exception as e:
            context.log.error(f"Model QA failed: {e}")
            mlflow.log_param("qa_gate_passed", "false_error")
            return {"status": "failed", "qa_passed": False, "error": str(e)}
