import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np

print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

# ────────────────────────────────────────────────
model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"\nLoading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,     # good choice for Apple Silicon
    device_map="auto",
)

print("Model loaded successfully!")

# You can check where layers are placed like this (optional):
print("First layer device:", next(model.parameters()).device)     # should show mps:0
print("Model dtype:", next(model.parameters()).dtype)            # should show torch.bfloat16

# ────────────────────────────────────────────────
# Slightly harder prompt to see more steps
prompt = """Think step by step and explain your reasoning clearly:

If it takes 5 machines 5 minutes to make 5 widgets,
how long would it take 100 machines to make 100 widgets?"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to("mps")

print("\nGenerating with hidden states...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        output_hidden_states=True,          # ← this is the new key line
        return_dict_in_generate=True
    )

generated_ids = outputs.sequences[0]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("\nGenerated text:\n")
print(generated_text)

# Quick look at hidden states
print("\nHidden states info:")
print(f"Number of generated tokens: {len(outputs.hidden_states)}")
print(f"Number of layers: {len(outputs.hidden_states[0])}")
print(f"Shape of hidden states for first generated token: {outputs.hidden_states[0][-1].shape}")
# usually (1, current_seq_len, hidden_size) — e.g. (1, 50, 3584) for Qwen2.5-7B

# Save last-layer hidden states for all generated tokens
last_layer_hidden = []
for step in outputs.hidden_states:
    last_layer = step[-1]               # last layer (index -1)
    last_token_state = last_layer[0, -1, :]   # last token of current sequence
    last_token_state.cpu().float().numpy()
    last_layer_hidden.append(last_token_state.cpu().float().numpy())

np.save("hidden_states_last_layer.npy", np.array(last_layer_hidden))
print("Saved last-layer hidden states to: hidden_states_last_layer.npy")
print(f"Saved {len(last_layer_hidden)} states, each of size {last_layer_hidden[0].shape}")


# After generation
generated_ids = outputs.sequences[0]
generated_tokens = generated_ids[len(inputs["input_ids"][0]):]  # only new tokens

last_layer_hidden = [step[-1][0, -1, :].cpu().float().numpy() for step in outputs.hidden_states]

# Very simple sentence-end detection (token level)
sentence_end_indices = []
for i, token_id in enumerate(generated_tokens):
    token_str = tokenizer.convert_ids_to_tokens(token_id.item())
    if token_str in [".", "!", "?", ".\n", "!\n", "?\n", "</s>"]:
        sentence_end_indices.append(i)

print("Detected possible sentence ends at token positions:", sentence_end_indices)

# After generation
input_len = inputs.input_ids.shape[1]
generated_text = tokenizer.decode(generated_ids[input_len:], skip_special_tokens=True)

# Find approximate positions of sentence ends in the decoded text
import re
sentence_boundaries = [m.end() for m in re.finditer(r'[.!?]\s+', generated_text + " ")]

print("Sentence end character positions in generated text:", sentence_boundaries)

# Optional: show the sentences
sentences = re.split(r'(?<=[.!?])\s+', generated_text.strip())
print("\nDetected sentences:")
for i, s in enumerate(sentences, 1):
    print(f"{i}. {s}")

# Save only those
if sentence_end_indices:
    sentence_states = [last_layer_hidden[i] for i in sentence_end_indices]
    np.save("sentence_final_states.npy", np.array(sentence_states))
    print(f"Saved {len(sentence_states)} sentence-final states")
