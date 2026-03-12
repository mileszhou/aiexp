from llama_cpp import Llama

# Load model (adjust path to your GGUF file)
llm = Llama(
    model_path="/Users/miles/.lmstudio/models/lmstudio-community/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf",
    n_gpu_layers=-1,          # -1 = full offload to GPU (Metal/CUDA); 0 = CPU only
    n_ctx=16384,              # context size (bump to 32k–128k+ on your hardware)
    n_threads=16,             # CPU threads if partial offload
    flash_attn=True,          # often faster on recent hardware
    verbose=False             # see loading/timings
)

# Simple completion
output = llm(
    "Hello! Tell me a short story about a robot learning to love coffee.",
    max_tokens=300,
    temperature=0.85,
    top_p=0.9,
    stop=["</s>", "\n\n"],    # optional stop tokens
    echo=True                 # include prompt in output
)

print("Full output: ", output)
print(output['choices'][0]['text'])