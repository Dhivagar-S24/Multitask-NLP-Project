from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai-community/gpt2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading GPT-2 model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("GPT-2 loaded successfully!")

prompt = "Artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nPrompt:")
print(prompt)

print("\nGenerated text:")
print(result)