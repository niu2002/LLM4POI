# AMD Remote Runbook

This branch is for a Linux AMD remote server with large VRAM. It keeps the v2 path on `ms-swift` for training and `vLLM` for serving, instead of the Windows local smoke helpers used on `windows_base`.

## Environment Baseline

The AMD machine is a ModelScope / DSW ROCm image. Treat the image runtime as the source of truth. Do not rebuild the environment from `v2/environment.yml`, and do not casually reinstall `torch`, `torchaudio`, or `torchvision`, because that can break the ROCm stack that is already on the machine.

From the environment snapshot you provided, the AMD box already has the core project stack around:

- `agentscope==0.1.6`
- `loguru==0.6.0`
- `accelerate==1.12.0`
- `datasets==3.6.0`
- `fastapi==0.128.0`
- `modelscope==1.37.1`
- `ms-swift==3.11.3`
- `openai==2.15.0`
- `pandas==2.2.3`
- `peft==0.18.1`
- `pyarrow==22.0.0`
- `transformers==4.55.4`
- `trl==0.22.2`
- `uvicorn==0.40.0`

If the live machine differs from this snapshot, prefer the live machine. The goal on AMD is "add missing project packages only", not "force the local environment to match a static lock file".

## What Changed From `windows_base`

- Removed Windows-only local smoke helpers from the AMD branch:
  - `v2/local_openai_server.py`
  - `v2/smoke_train_local.py`
- Kept the script syntax fixes and environment-variable based launch style.
- Raised remote defaults to use more memory and reduced output noise:
  - `v2/sft.sh`: `MAX_LENGTH=32768`
  - `v2/serve_vllm.sh`: `MAX_MODEL_LEN=32768`
  - `v2/serve_vllm.sh`: `MAX_NUM_BATCHED_TOKENS=131072`
  - `v2/serve_vllm.sh`: `MAX_NUM_SEQS=128`
  - `v2/serve_vllm.sh`: `GPU_MEMORY_UTILIZATION=0.9`
  - `v2/sft.sh`: `LOGGING_STRATEGY=epoch`
  - `v2/sft.sh`: `SAVE_STRATEGY=epoch`
  - `v2/sft.sh`: `SAVE_TOTAL_LIMIT=1`
  - `v2/sft.sh`: `DISABLE_TQDM=true`
  - `v2/sft.sh`: `REPORT_TO=none`
  - `v2/sft.sh`: `QUIET_OUTPUT=1`
  - `v2/sft.sh`: use the active environment's `python -m torch.distributed.run`

The default console output now keeps epoch-level averages, final runtime/checkpoint
information, and errors. Set `QUIET_OUTPUT=0` only when full framework diagnostics
are needed. These defaults are intended as a starting point for a 192 GB AMD GPU.
If the model is small and memory remains low, raise
`PER_DEVICE_TRAIN_BATCH_SIZE`, `MAX_LENGTH`, or both.

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

## Install Project Dependencies Safely

The old `ops/install_amd_deps.sh` idea came from an earlier project flow and is deprecated here. Do not rely on that old script name on the AMD server.

After `git pull`, install only the project-level dependencies that are actually missing. Keep the existing ROCm / PyTorch stack in place:

```bash
pip install --upgrade-strategy only-if-needed \
  agentscope==0.1.6 \
  loguru==0.6.0 \
  accelerate==1.12.0 \
  datasets==3.6.0 \
  fastapi==0.128.0 \
  modelscope==1.37.1 \
  ms-swift==3.11.3 \
  openai==2.15.0 \
  pandas==2.2.3 \
  peft==0.18.1 \
  pyarrow==22.0.0 \
  transformers==4.55.4 \
  trl==0.22.2 \
  uvicorn==0.40.0
```

This environment repair should intentionally avoid reinstalling:

- `torch`
- `torchaudio`
- `torchvision`

If you later find that `vllm`, `deepspeed`, or another package needs a ROCm-specific build, install that one package deliberately after checking the current environment, instead of bulk reinstalling everything.

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

On ROCm, `vLLM` startup is slow enough that it is easy to misdiagnose as a failure. The first launch of a model such as `Llama-3.1-8B-Instruct` may take more than 80 seconds because the server still needs to load weights, compile kernels, warm up, and capture graphs. A failed `curl` after `sleep 30` or `sleep 40` is not enough to conclude that serving failed.

Example startup:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 7863 \
  --model /mnt/workspace/comapoilatest/models/Llama-3.1-8B-Instruct \
  --served-model-name llama3.1-8b \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.85
```

Recommended health check:

```bash
curl -m 5 -sS http://127.0.0.1:7863/v1/models
```

If the service is ready, the response should include `llama3.1-8b`.

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
MODEL_NAME=llm4poi-lora \
MAX_NEW_TOKENS=48 \
CONCURRENCY=32 \
TEMPERATURE=0.0 \
bash eval.sh
```

If you really need per-sample prediction details, enable them explicitly:

```bash
WRITE_DETAILS=1 \
OUTPUT_PATH=../outputs/nyc_amd_remote_predictions.jsonl \
bash eval.sh
```

If the model is still not filling VRAM during evaluation, increase `CONCURRENCY`. If vLLM reports memory pressure, reduce `MAX_NUM_SEQS` or `MAX_NUM_BATCHED_TOKENS`.

## Generative Hit@K Evaluation

The default `v2/eval.py` reports single-sample exact match, which is equivalent to `Hit@1` for the current generative v2 path. To estimate `Hit@1/5/10/20`, use the multi-sample evaluator:

```bash
cd v2

DATASET_PATH=../datasets/nyc/llm4poi_v2/nyc_gsm8k_test_llm4poi.parquet \
MODEL_NAME=llm4poi \
BASE_URL=http://127.0.0.1:7864/v1 \
MAX_NEW_TOKENS=48 \
CONCURRENCY=8 \
TEMPERATURE=0.7 \
TOP_P=0.95 \
NUM_RETURN_SEQUENCES=20 \
K_VALUES=1,5,10,20 \
bash eval_hitk.sh
```

To save detailed sampled predictions:

```bash
WRITE_DETAILS=1 \
OUTPUT_PATH=../outputs/nyc_hitk_predictions.jsonl \
bash eval_hitk.sh
```

## Cleanup Historical Pure Logs

Preview removable log-like files first:

```bash
bash ops/cleanup_large_outputs.sh ../outputs
```

Delete them after preview:

```bash
PREVIEW_ONLY=0 bash ops/cleanup_large_outputs.sh ../outputs
```

The cleanup script only removes logs and detailed evaluation JSONL files. It
does not delete checkpoints. In the current 45 GB workspace, most disk usage is
model weights rather than logs. After confirming that `checkpoint-64` is the
required final model, the known historical checkpoints can be removed manually:

```bash
rm -rf \
  /mnt/workspace/LLM4POI/outputs/nyc_sft_smoke_100 \
  /mnt/workspace/LLM4POI/outputs/nyc_sft_500_full/v1-20260611-132726/checkpoint-50
```

This retains:

```text
/mnt/workspace/LLM4POI/outputs/nyc_sft_500_full/v1-20260611-132726/checkpoint-64
```

This is a generative `Hit@K`: each request asks the model for multiple sampled answers and checks whether the gold POI id appears in the first `K` generations. It is useful for the v2 serving path, but it is not the same as a full candidate-pool logprob reranker.

## Model Size Guidance

With 192 GB VRAM, you should be able to run a much larger model than the local 2B Windows test model, assuming ROCm, PyTorch, ms-swift, and vLLM are installed correctly. Send the remote model path and model type when ready; the main variables to tune will be:

- `MODEL_PATH`
- `MODEL_TYPE`
- `NPROC_PER_NODE`
- `PER_DEVICE_TRAIN_BATCH_SIZE`
- `MAX_LENGTH`
- `TENSOR_PARALLEL_SIZE`
