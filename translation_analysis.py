from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==========================================
# MULTILINGUAL TRANSLATION - QUALITATIVE TEST
# ==========================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Loading multilingual translation model...")
print("If the model is already downloaded, it will load from cache.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Translation model loaded successfully!\n")


# Language codes used by NLLB
LANG_CODES = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu"
}


def translate(text, source, target):

    tokenizer.src_lang = LANG_CODES[source]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True
    )

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            LANG_CODES[target]
        ),
        max_new_tokens=100
    )

    result = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return result


# ==========================================
# TEST DATA
# ==========================================

tests = [

    # English -> Tamil
    {
        "source": "en",
        "target": "ta",
        "text": "Artificial intelligence is changing the world."
    },

    # English -> Telugu
    {
        "source": "en",
        "target": "te",
        "text": "Artificial intelligence is changing the world."
    },

    # Tamil -> English
    {
        "source": "ta",
        "target": "en",
        "text": "செயற்கை நுண்ணறிவு உலகத்தை மாற்றுகிறது."
    },

    # Telugu -> English
    {
        "source": "te",
        "target": "en",
        "text": "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది."
    },

    # Tamil -> Telugu
    {
        "source": "ta",
        "target": "te",
        "text": "செயற்கை நுண்ணறிவு உலகத்தை மாற்றுகிறது."
    },

    # Telugu -> Tamil
    {
        "source": "te",
        "target": "ta",
        "text": "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది."
    }
]


# ==========================================
# LANGUAGE NAMES
# ==========================================

language_names = {
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu"
}


# ==========================================
# RUN QUALITATIVE TESTS
# ==========================================

print("=" * 60)
print("QUALITATIVE TRANSLATION ANALYSIS")
print("=" * 60)

for i, test in enumerate(tests, 1):

    source = test["source"]
    target = test["target"]
    text = test["text"]

    print(f"\nTest {i}")
    print("-" * 60)

    print(
        f"Direction : "
        f"{language_names[source]} -> {language_names[target]}"
    )

    print(f"Source    : {text}")

    try:
        result = translate(text, source, target)

        print(f"Generated : {result}")

    except Exception as e:
        print(f"ERROR     : {e}")


print("\n" + "=" * 60)
print("QUALITATIVE ANALYSIS COMPLETED")
print("=" * 60)