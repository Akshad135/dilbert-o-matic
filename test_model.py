from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_DIR = "./models/t5_jargon_v1"

print(f"Loading model from {MODEL_DIR}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# UPDATED: Better instruction prefix for FLAN-T5
prefix = "Translate to corporate jargon: "

# Test examples
test_examples = [
    "I'm going to get coffee.",
    "Let's talk about this in a meeting.",
    "The project is delayed.",
    "We need to reduce costs.",
]

for simple_text in test_examples:
    print(f"\n{'='*50}")
    print(f"Input: '{simple_text}'")
    
    inputs = tokenizer(
        prefix + simple_text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    ).to(device)
    
    # UPDATED: Better generation parameters
    output_tokens = model.generate(
        **inputs,
        max_length=128,
        min_length=10,  # Prevent trivial outputs
        num_beams=4,  # Reduced from 5
        temperature=0.9,  # Added creativity
        do_sample=False,  # Deterministic with beam search
        early_stopping=True,
        no_repeat_ngram_size=2,  # Prevent repetition
    )
    
    jargon_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(f"Jargon: '{jargon_text}'")

print(f"\n{'='*50}\n")
