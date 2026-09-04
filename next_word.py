import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai-community/gpt2"

print("Loading GPT-2...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!\n")


def predict_next_words(prompt, top_k=5):
    # Convert prompt into tokens
    inputs = tokenizer(prompt, return_tensors="pt")

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions for the last token
    logits = outputs.logits[:, -1, :]

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Get top K predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        top_k
    )

    print("Prompt:")
    print(prompt)

    print("\nTop predictions:")

    results = []
    for i in range(top_k):
        token_id = top_indices[0][i].item()
        probability = top_probabilities[0][i].item()

        word = tokenizer.decode([token_id])

        print(
            f"{i + 1}. {word!r} "
            f"({probability * 100:.2f}%)"
        )
        results.append({
            "word": word,
            "probability": round(probability * 100, 2)
        })
    return results


if __name__ == "__main__":
    prompt = input("Enter a sentence: ")
    predict_next_words(prompt)