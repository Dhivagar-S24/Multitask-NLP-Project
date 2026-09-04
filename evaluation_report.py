import matplotlib.pyplot as plt

# ==========================================
# MULTITASK NLP - FINAL EVALUATION REPORT
# ==========================================

# -----------------------------
# 1. MODEL RESULTS
# -----------------------------

next_word_top1 = 20.00
next_word_top5 = 40.00

test_loss = 3.0578
perplexity = 21.2812

bleu_scores = {
    "EN → TA": 44.05,
    "EN → TE": 60.33,
    "TA → EN": 68.98,
    "TE → EN": 88.30,
    "TA → TE": 53.83,
    "TE → TA": 49.93
}


# -----------------------------
# 2. PRINT FINAL REPORT
# -----------------------------

print("=" * 60)
print("        MULTITASK NLP FINAL EVALUATION")
print("=" * 60)

print("\nNEXT-WORD PREDICTION")
print("-" * 60)
print(f"Top-1 Accuracy : {next_word_top1:.2f}%")
print(f"Top-5 Accuracy : {next_word_top5:.2f}%")

print("\nPERPLEXITY")
print("-" * 60)
print(f"Test Loss      : {test_loss:.4f}")
print(f"Perplexity     : {perplexity:.4f}")

print("\nTRANSLATION BLEU SCORES")
print("-" * 60)

for direction, score in bleu_scores.items():
    print(f"{direction:<10}: {score:.2f}")

print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETED")
print("=" * 60)


# -----------------------------
# 3. BLEU SCORE GRAPH
# -----------------------------

directions = list(bleu_scores.keys())
scores = list(bleu_scores.values())

plt.figure(figsize=(10, 6))

plt.bar(directions, scores)

plt.title("Multilingual Translation BLEU Scores")
plt.xlabel("Translation Direction")
plt.ylabel("BLEU Score")

plt.ylim(0, 100)

for i, score in enumerate(scores):
    plt.text(i, score + 1, f"{score:.2f}", ha="center")

plt.tight_layout()

plt.savefig("bleu_scores.png", dpi=300)

print("\nBLEU graph saved as: bleu_scores.png")


# -----------------------------
# 4. NEXT-WORD ACCURACY GRAPH
# -----------------------------

accuracy_names = [
    "Top-1 Accuracy",
    "Top-5 Accuracy"
]

accuracy_values = [
    next_word_top1,
    next_word_top5
]

plt.figure(figsize=(8, 6))

plt.bar(accuracy_names, accuracy_values)

plt.title("Next-Word Prediction Accuracy")
plt.xlabel("Evaluation Metric")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

for i, value in enumerate(accuracy_values):
    plt.text(i, value + 2, f"{value:.2f}%", ha="center")

plt.tight_layout()

plt.savefig("next_word_accuracy.png", dpi=300)

print("Accuracy graph saved as: next_word_accuracy.png")


# -----------------------------
# 5. SAVE TEXT REPORT
# -----------------------------

with open("final_evaluation_report.txt", "w", encoding="utf-8") as file:

    file.write("MULTITASK NLP FINAL EVALUATION REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write("NEXT-WORD PREDICTION\n")
    file.write(f"Top-1 Accuracy: {next_word_top1:.2f}%\n")
    file.write(f"Top-5 Accuracy: {next_word_top5:.2f}%\n\n")

    file.write("PERPLEXITY\n")
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Perplexity: {perplexity:.4f}\n\n")

    file.write("TRANSLATION BLEU SCORES\n")

    for direction, score in bleu_scores.items():
        file.write(f"{direction}: {score:.2f}\n")

print("Text report saved as: final_evaluation_report.txt")

print("\nAll evaluation files created successfully!")