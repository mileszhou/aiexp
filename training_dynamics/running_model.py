import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"           # or "google/gemma-2-9b-it", etc.

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 4-bit quantization to save memory
# (you can also use device_map="auto", torch_dtype=torch.bfloat16 without quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",   # faster if you have recent macOS + metal
    # load_in_4bit=True, quantization_config=BitsAndBytesConfig(...)   # if you want even lower memory
)

# Enable gradient checkpointing if memory gets tight
model.gradient_checkpointing_enable()

print("Model loaded on:", model.hf_device_map)

# Example inference
prompt = """Solve this step by step:

A farmer has 17 sheep. All but 9 die. How many sheep are left?
"""

messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(input_text, return_tensors="pt").to("mps")   # or "cuda" / "cpu"

# Generate with hidden states
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,           # greedy for cleanest traces
        output_hidden_states=True, # <--- this is the key
        return_dict_in_generate=True
    )

# outputs.hidden_states is a tuple of length = number of generated tokens
# Each element: tuple of length = number of layers
# outputs.hidden_states[step][layer] has shape (1, seq_len_so_far, hidden_size)
print("Generated tokens:", tokenizer.batch_decode(outputs.sequences)[0])
print("Hidden states for each step and layer:")
for step, step_hidden in enumerate(outputs.hidden_states):
    print(f"Step {step}:")
    for layer, layer_hidden in enumerate(step_hidden):
        print(f"  Layer {layer}: shape {layer_hidden.shape}")   

