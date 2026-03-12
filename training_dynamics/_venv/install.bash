# Installation of training dynamics dependencies
pip install transformers torch accelerate bitsandbytes sentencepiece protobuf
# optional but helpful:
pip install huggingface_hub hf_transfer

# python
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
    attn_implementation="flash_attention_2",   # faster if you have recent macOS + metal
    # load_in_4bit=True, quantization_config=BitsAndBytesConfig(...)   # if you want even lower memory
)

# Enable gradient checkpointing if memory gets tight
model.gradient_checkpointing_enable()

print("Model loaded on:", model.hf_device_map)
