import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai-community/gpt2"

print("Loading GPT-2...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!\n")


# Test sentences
test_sentences = [
    "Artificial intelligence is changing",
    "Machine learning is a",
    "Deep learning uses neural",
    "Natural language processing allows",
    "Large language models can",
    "The future of technology is",
    "Python is a popular",
    "Transformers are used for",
    "Generative AI can create",
    "Computer science is an"
]


top1_correct = 0
top5_correct = 0
total = 0


for sentence in test_sentences:

    # Tokenize complete sentence
    tokens = tokenizer(
        sentence,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"]

    # Use all tokens except the last one as input
    input_ids_without_last = input_ids[:, :-1]

    # The last token is the actual answer
    actual_token = input_ids[:, -1].item()

    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_without_last
        )

    # Predictions for the next token
    logits = outputs.logits[:, -1, :]

    probabilities = torch.softmax(logits, dim=-1)

    # Get top 5
    top_probabilities, top_indices = torch.topk(
        probabilities,
        5
    )

    predictions = top_indices[0].tolist()

    # Top-1
    if actual_token == predictions[0]:
        top1_correct += 1

    # Top-5
    if actual_token in predictions:
        top5_correct += 1

    total += 1

    actual_word = tokenizer.decode([actual_token])

    predicted_words = [
        tokenizer.decode([token_id])
        for token_id in predictions
    ]

    print("----------------------------------------")
    print("Sentence:", sentence)
    print("Actual next token:", repr(actual_word))
    print("Top-5 predictions:", predicted_words)


top1_accuracy = (top1_correct / total) * 100
top5_accuracy = (top5_correct / total) * 100


print("\n========================================")
print("NEXT-WORD PREDICTION EVALUATION")
print("========================================")

print(f"Total test samples: {total}")
print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")