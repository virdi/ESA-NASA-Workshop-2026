# Use the official vLLM image with CUDA support as the base
# (already packages Python 3.12, uv, vLLM, CUDA toolkit, TIMM, and more)
FROM vllm/vllm-openai:v0.18.0

# Install terratorch with vLLM support at version 1.2.5
RUN pip3 install terratorch[vllm]==1.2.5