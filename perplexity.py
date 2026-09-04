import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai-community/gpt2"

print("Loading GPT-2...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

print("GPT-2 loaded successfully!\n")


# Test corpus
test_text = """
Artificial intelligence is transforming many industries.
Machine learning allows computers to learn patterns from data.
Deep learning uses neural networks with multiple layers.
Natural language processing enables computers to understand human language.
Large language models can generate and understand text.
Generative artificial intelligence can create new content.
Transformers are widely used in modern language processing.
"""


def evaluate_perplexity(text=None):
    if text is None or not text.strip():
        eval_text = test_text
    else:
        eval_text = text

    encodings = tokenizer(
        eval_text,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=input_ids
        )

    loss = outputs.loss.item()
    perplexity = math.exp(loss)

    return {
        "loss": round(loss, 4),
        "perplexity": round(perplexity, 4)
    }


if __name__ == "__main__":
    encodings = tokenizer(test_text, return_tensors="pt")
    input_ids = encodings["input_ids"]
    print("Number of tokens:", input_ids.size(1))

    res = evaluate_perplexity(test_text)
    print("\n==============================")
    print("PERPLEXITY EVALUATION")
    print("==============================")

    print(f"Test Loss: {res['loss']:.4f}")
    print(f"Perplexity: {res['perplexity']:.4f}")