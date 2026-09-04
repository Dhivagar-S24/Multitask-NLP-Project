from transformers import pipeline

print("Loading sentiment analysis model...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Model loaded successfully!")

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return {
        "sentiment": result["label"],
        "confidence": round(result["score"] * 100, 2)
    }


if __name__ == "__main__":
    print("\n======================================")
    print("       SENTIMENT ANALYSIS")
    print("======================================")

    # Test sentences
    test_sentences = [
        "I love this project.",
        "This application is amazing.",
        "The system works very well.",
        "I do not like this application.",
        "This project is terrible and disappointing.",
        "The system is slow and frustrating.",
        "The results are very poor.",
        "Artificial intelligence is changing the world."
    ]

    print("\nTEST RESULTS")
    print("--------------------------------------")

    for i, sentence in enumerate(test_sentences, 1):
        res = analyze_sentiment(sentence)
        print(f"\nTest {i}")
        print(f"Sentence    : {sentence}")
        print(f"Sentiment   : {res['sentiment']}")
        print(f"Confidence  : {res['confidence']:.2f}%")

    # Interactive mode
    print("\n======================================")
    print("     INTERACTIVE SENTIMENT ANALYSIS")
    print("======================================")

    print("Enter a sentence to analyze.")
    print("Type 'exit' to stop.")

    while True:
        text = input("\nEnter sentence: ")

        if text.lower() == "exit":
            print("\nSentiment analysis stopped.")
            break

        if not text.strip():
            print("Please enter a sentence.")
            continue

        res = analyze_sentiment(text)
        print("\nResult")
        print("--------------------------------------")
        print(f"Sentence   : {text}")
        print(f"Sentiment  : {res['sentiment']}")
        print(f"Confidence : {res['confidence']:.2f}%")