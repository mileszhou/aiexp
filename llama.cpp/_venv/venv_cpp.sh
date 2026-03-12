#!/usr/bin/env bash
set -e

python3 -m venv llama.cpp/.venv.cpp
source llama.cpp/.venv.cpp/bin/activate
    
PATH=llama.cpp/.venv.cpp/bin:$PATH

pip install --upgrade pip
# Forces Metal backend + often faster builds
pip install llama-cpp-python --force-reinstall --no-cache-dir \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
