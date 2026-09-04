from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu


MODEL_NAME = "facebook/nllb-200-distilled-600M"


# NLLB language codes
LANGUAGES = {
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu"
}


print("=" * 50)
print("MACHINE TRANSLATION - BLEU EVALUATION")
print("=" * 50)

print("\nLoading NLLB-200 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!\n")


def translate(text, source, target):

    tokenizer.src_lang = LANGUAGES[source]

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            LANGUAGES[target]
        ),
        max_new_tokens=100
    )

    result = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return result


# =========================================================
# TEST DATA
# =========================================================

data = {

    "en-ta": {
        "source": [
            "Artificial intelligence is changing the world.",
            "I am studying computer science.",
            "Technology makes life easier.",
            "Machine learning is a part of artificial intelligence.",
            "Students use computers for learning."
        ],

        "reference": [
            "செயற்கை நுண்ணறிவு உலகத்தை மாற்றி வருகிறது.",
            "நான் கணினி அறிவியல் படித்து வருகிறேன்.",
            "தொழில்நுட்பம் வாழ்க்கையை எளிதாக்குகிறது.",
            "இயந்திரக் கற்றல் செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும்.",
            "மாணவர்கள் கற்றுக்கொள்ள கணினிகளைப் பயன்படுத்துகிறார்கள்."
        ]
    },


    "en-te": {
        "source": [
            "Artificial intelligence is changing the world.",
            "I am studying computer science.",
            "Technology makes life easier.",
            "Machine learning is a part of artificial intelligence.",
            "Students use computers for learning."
        ],

        "reference": [
            "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
            "నేను కంప్యూటర్ సైన్స్ చదువుతున్నాను.",
            "సాంకేతికత జీవితాన్ని సులభతరం చేస్తుంది.",
            "మెషిన్ లెర్నింగ్ కృత్రిమ మేధస్సులో ఒక భాగం.",
            "విద్యార్థులు నేర్చుకోవడానికి కంప్యూటర్లను ఉపయోగిస్తారు."
        ]
    },


    "ta-en": {
        "source": [
            "செயற்கை நுண்ணறிவு உலகத்தை மாற்றி வருகிறது.",
            "நான் கணினி அறிவியல் படித்து வருகிறேன்.",
            "தொழில்நுட்பம் வாழ்க்கையை எளிதாக்குகிறது.",
            "இயந்திரக் கற்றல் செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும்.",
            "மாணவர்கள் கற்றுக்கொள்ள கணினிகளைப் பயன்படுத்துகிறார்கள்."
        ],

        "reference": [
            "Artificial intelligence is changing the world.",
            "I am studying computer science.",
            "Technology makes life easier.",
            "Machine learning is a part of artificial intelligence.",
            "Students use computers for learning."
        ]
    },


    "te-en": {
        "source": [
            "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
            "నేను కంప్యూటర్ సైన్స్ చదువుతున్నాను.",
            "సాంకేతికత జీవితాన్ని సులభతరం చేస్తుంది.",
            "మెషిన్ లెర్నింగ్ కృత్రిమ మేధస్సులో ఒక భాగం.",
            "విద్యార్థులు నేర్చుకోవడానికి కంప్యూటర్లను ఉపయోగిస్తారు."
        ],

        "reference": [
            "Artificial intelligence is changing the world.",
            "I am studying computer science.",
            "Technology makes life easier.",
            "Machine learning is a part of artificial intelligence.",
            "Students use computers for learning."
        ]
    },


    "ta-te": {
        "source": [
            "செயற்கை நுண்ணறிவு உலகத்தை மாற்றி வருகிறது.",
            "நான் கணினி அறிவியல் படித்து வருகிறேன்.",
            "தொழில்நுட்பம் வாழ்க்கையை எளிதாக்குகிறது.",
            "இயந்திரக் கற்றல் செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும்.",
            "மாணவர்கள் கற்றுக்கொள்ள கணினிகளைப் பயன்படுத்துகிறார்கள்."
        ],

        "reference": [
            "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
            "నేను కంప్యూటర్ సైన్స్ చదువుతున్నాను.",
            "సాంకేతికత జీవితాన్ని సులభతరం చేస్తుంది.",
            "మెషిన్ లెర్నింగ్ కృత్రిమ మేధస్సులో ఒక భాగం.",
            "విద్యార్థులు నేర్చుకోవడానికి కంప్యూటర్లను ఉపయోగిస్తారు."
        ]
    },


    "te-ta": {
        "source": [
            "కృత్రిమ మేధస్సు ప్రపంచాన్ని మారుస్తోంది.",
            "నేను కంప్యూటర్ సైన్స్ చదువుతున్నాను.",
            "సాంకేతికత జీవితాన్ని సులభతరం చేస్తుంది.",
            "మెషిన్ లెర్నింగ్ కృత్రిమ మేధస్సులో ఒక భాగం.",
            "విద్యార్థులు నేర్చుకోవడానికి కంప్యూటర్లను ఉపయోగిస్తారు."
        ],

        "reference": [
            "செயற்கை நுண்ணறிவு உலகத்தை மாற்றி வருகிறது.",
            "நான் கணினி அறிவியல் படித்து வருகிறேன்.",
            "தொழில்நுட்பம் வாழ்க்கையை எளிதாக்குகிறது.",
            "இயந்திரக் கற்றல் செயற்கை நுண்ணறிவின் ஒரு பகுதியாகும்.",
            "மாணவர்கள் கற்றுக்கொள்ள கணினிகளைப் பயன்படுத்துகிறார்கள்."
        ]
    }
}


# =========================================================
# EVALUATION
# =========================================================

results = {}


for direction, values in data.items():

    source_lang = direction.split("-")[0]
    target_lang = direction.split("-")[1]

    print("\n" + "=" * 50)
    print(f"{source_lang.upper()} -> {target_lang.upper()}")
    print("=" * 50)

    predictions = []

    for i, sentence in enumerate(values["source"]):

        prediction = translate(
            sentence,
            source_lang,
            target_lang
        )

        predictions.append(prediction)

        print(f"\nTest {i + 1}")
        print("Source    :", sentence)
        print("Generated :", prediction)
        print("Reference :", values["reference"][i])

    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(
        predictions,
        [values["reference"]]
    )

    results[direction] = bleu.score

    print(f"\nBLEU Score: {bleu.score:.2f}")


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n")
print("=" * 50)
print("FINAL BLEU RESULTS")
print("=" * 50)

for direction, score in results.items():

    source, target = direction.split("-")

    print(
        f"{source.upper()} -> {target.upper()} : "
        f"{score:.2f}"
    )

print("=" * 50)