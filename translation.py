from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading multilingual translation model...")
print("This model is about 2.5 GB, so the first download may take some time.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Translation model loaded successfully!\n")


# NLLB language codes
LANGUAGES = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu"
}


def translate(text, source, target):

    source_lang = LANGUAGES[source]
    target_lang = LANGUAGES[target]

    # Set source language
    tokenizer.src_lang = source_lang

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    # Generate translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_new_tokens=100
    )

    # Convert tokens to text
    result = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return result


if __name__ == "__main__":
    print("======================================")
    print("   MULTILINGUAL TRANSLATION SYSTEM")
    print("======================================")

    print("""
Supported languages:

en = English
ta = Tamil
te = Telugu

Supported directions:

English → Tamil
English → Telugu
Tamil → English
Tamil → Telugu
Telugu → English
Telugu → Tamil
""")

    while True:

        source = input("\nSource language (en/ta/te): ")

        if source.lower() == "exit":
            break

        target = input("Target language (en/ta/te): ")

        if target.lower() == "exit":
            break

        if source.lower() not in LANGUAGES or target.lower() not in LANGUAGES:
            print("Invalid language code!")
            continue

        if source.lower() == target.lower():
            print("Source and target languages cannot be the same.")
            continue

        text = input("Enter text: ")

        print("\nTranslating...")

        result = translate(
            text,
            source.lower(),
            target.lower()
        )

        print("\n--------------------------------------")
        print("Translation:")
        print(result)
        print("--------------------------------------")