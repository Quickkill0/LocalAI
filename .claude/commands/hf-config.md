# Generate LocalAI Config from HuggingFace Model

Generate a LocalAI YAML configuration file from a HuggingFace model URL.

## Input
$ARGUMENTS - HuggingFace model URL (e.g., https://huggingface.co/TheBloke/Llama-2-7B-GGUF)

## Instructions

1. Parse the HuggingFace URL to extract the repo ID (e.g., "TheBloke/Llama-2-7B-GGUF")

2. Fetch the model details from HuggingFace API:
   - List all files in the repository
   - Identify GGUF files (for llama-cpp backend)
   - Identify safetensors files (for transformers/vllm backend)
   - Check for mmproj files (vision models)
   - Get SHA256 checksums from LFS metadata

3. Detect the model type and appropriate backend:
   - **GGUF files** → `backend: llama-cpp`
   - **Safetensors with "diffusion"** → `backend: diffusers`
   - **Safetensors LLM** → `backend: vllm` or `backend: transformers`
   - **Whisper models** → `backend: whisper`
   - **Piper/TTS models** → `backend: piper`

4. For GGUF models, identify available quantizations and recommend:
   - Q4_K_M (good balance of quality/size)
   - Q5_K_M (better quality)
   - Q8_0 (high quality)
   - Show all available options to user

5. Generate the YAML config with:
   - Appropriate backend
   - Sensible defaults (context_size, mmap, f16, etc.)
   - Chat/completion templates if detectable from model card
   - download_files with proper URIs and SHA256

6. Save the config to `shared-files/` and provide download link

## Example Output Format

```yaml
name: "llama-2-7b-q4km"
backend: llama-cpp
context_size: 4096
f16: true
mmap: true

parameters:
  model: llama-2-7b.Q4_K_M.gguf

template:
  chat_message: |
    [INST] {{.Content}} [/INST]
  chat: |
    {{.Input}}

stopwords:
- "</s>"
- "[INST]"
- "[/INST]"

download_files:
- filename: llama-2-7b.Q4_K_M.gguf
  sha256: abc123...
  uri: huggingface://TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf
```

## Template Detection Hints

Look for these patterns in model names/cards:
- **Llama 2/3**: `[INST]...[/INST]` format
- **ChatML**: `<|im_start|>...<|im_end|>` format
- **Alpaca**: `### Instruction:...### Response:` format
- **Vicuna**: `USER:...ASSISTANT:` format
- **Mistral**: Similar to Llama 2 instruct format
- **Phi**: `<|user|>...<|assistant|>` format

If unsure, use `use_tokenizer_template: true` for vllm/transformers backends.
