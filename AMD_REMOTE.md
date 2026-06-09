# AMD Remote Runbook

This branch is for a Linux AMD remote server with large VRAM. It keeps the v2 path on `ms-swift` for training and `vLLM` for serving, instead of the Windows local smoke helpers used on `windows_base`.

## What Changed From `windows_base`

- Removed Windows-only local smoke helpers from the AMD branch:
  - `v2/local_openai_server.py`
  - `v2/smoke_train_local.py`
- Kept the script syntax fixes and environment-variable based launch style.
- Raised remote defaults to use more memory:
  - `v2/sft.sh`: `MAX_LENGTH=32768`
  - `v2/serve_vllm.sh`: `MAX_MODEL_LEN=32768`
  - `v2/serve_vllm.sh`: `MAX_NUM_BATCHED_TOKENS=131072`
  - `v2/serve_vllm.sh`: `MAX_NUM_SEQS=128`
  - `v2/serve_vllm.sh`: `GPU_MEMORY_UTILIZATION=0.9`

These defaults are intended as a starting point for a 192 GB AMD GPU. If the model is small and memory remains low, raise `PER_DEVICE_TRAIN_BATCH_SIZE`, `MAX_LENGTH`, or both.

## Pull This Branch On The AMD Server

```bash
git clone https://github.com/niu2002/LLM4POI.git
cd LLM4POI
git fetch origin
git checkout amd_remote
```

If the repo already exists:

```bash
cd LLM4POI
git fetch origin
git checkout amd_remote
git pull --ff-only
```

## No-GPU / Dry-Run Checks

Use these first to verify paths, environment variables, and command construction without loading a model.

```bash
cd v2

MODEL_PATH=/path/to/model \
MODEL_TYPE=qwen2_5 \
DATASET_PATH=/path/to/train.parquet \
OUTPUT_DIR=/path/to/output \
DRY_RUN=1 \
bash sft.sh

MODEL_DIR=/path/to/model \
DRY_RUN=1 \
bash serve_vllm.sh
```

## Data Preparation

Download the dataset zip from the README and unpack the target dataset. Then run the v2 converter:

```bash
python v2/convert_prompt_llm4poi.py \
  --dataset nyc \
  --train_csv datasets/nyc/preprocessed/train_sample.csv \
  --test_csv datasets/nyc/preprocessed/test_sample_with_traj.csv \
  --out_dir datasets/nyc/llm4poi_v2 \
  --history_limit 50 \
  --write_jsonl
```

The default converter writes parquet files:

```text
datasets/nyc/llm4poi_v2/nyc_gsm8k_train_llm4poi.parquet
datasets/nyc/llm4poi_v2/nyc_gsm8k_test_llm4poi.parquet
```

## High-Memory Training

Start from this. It preserves the original v2 spirit (`full` training, `zero3`, `all-linear`) while using a longer context.

```bash
cd v2

MODEL_PATH=/path/to/model \
MODEL_TYPE=qwen2_5 \
DATASET_PATH=../datasets/nyc/llm4poi_v2/nyc_gsm8k_train_llm4poi.parquet \
OUTPUT_DIR=../outputs/nyc_amd_remote_full \
NPROC_PER_NODE=1 \
MAX_LENGTH=32768 \
PER_DEVICE_TRAIN_BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=8 \
TORCH_DTYPE=bfloat16 \
DEEPSPEED_CONFIG=zero3 \
bash sft.sh
```

For a larger model or lower memory headroom, reduce batch size first:

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=8
```

For a smaller model and unused VRAM, raise batch size first:

```bash
PER_DEVICE_TRAIN_BATCH_SIZE=24
```

Then consider:

```bash
MAX_LENGTH=49152
MAX_NUM_BATCHED_TOKENS=196608
```

## Serving With vLLM

Full checkpoint:

```bash
cd v2

MODEL_DIR=/path/to/full-checkpoint-or-base-model \
SERVE_NAME=llm4poi \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=128 \
MAX_NUM_BATCHED_TOKENS=131072 \
GPU_MEMORY_UTILIZATION=0.9 \
bash serve_vllm.sh
```

LoRA adapter:

```bash
MODEL_DIR=/path/to/base-model \
ENABLE_LORA=1 \
ADAPTER_DIR=../outputs/nyc_amd_remote_full \
SERVE_NAME=llm4poi-lora \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=128 \
MAX_NUM_BATCHED_TOKENS=131072 \
GPU_MEMORY_UTILIZATION=0.9 \
bash serve_vllm.sh
```

## Evaluation

```bash
cd v2

DATASET_PATH=../datasets/nyc/llm4poi_v2/nyc_gsm8k_test_llm4poi.parquet \
OUTPUT_PATH=../outputs/nyc_amd_remote_predictions.jsonl \
MODEL_NAME=llm4poi-lora \
MAX_NEW_TOKENS=48 \
CONCURRENCY=32 \
TEMPERATURE=0.0 \
bash eval.sh
```

If the model is still not filling VRAM during evaluation, increase `CONCURRENCY`. If vLLM reports memory pressure, reduce `MAX_NUM_SEQS` or `MAX_NUM_BATCHED_TOKENS`.

## Model Size Guidance

With 192 GB VRAM, you should be able to run a much larger model than the local 2B Windows test model, assuming ROCm, PyTorch, ms-swift, and vLLM are installed correctly. Send the remote model path and model type when ready; the main variables to tune will be:

- `MODEL_PATH`
- `MODEL_TYPE`
- `NPROC_PER_NODE`
- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `MAX_LENGTH`
- `TENSOR_PARALLEL_SIZE`
