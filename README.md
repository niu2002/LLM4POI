# Large Language Models for Next Point-of-Interest Recommendation
[![License: APACHE-2.0](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://www.apache.org/licenses/LICENSE-2.0)
[![Venue:SIGIR 2024](https://img.shields.io/badge/Venue-SIGIR2024-orange)](https://sigir-2024.github.io/index.html)

This repository includes the implementation of the paper "[Large Language Models for Next Point-of-Interest Recommendation](https://arxiv.org/pdf/2404.17591)".

**Please select the version you wish to use (we strongly recommend you try the v2 implementation):**
---

<details open>
<summary>v2: swift based training</summary>
<br>

> **Note:** This is the latest version of the framework.

### Install
1. Clone this repository to your local machine.
2. Install the environment or follow the instructions from [ms-swift](https://github.com/modelscope/ms-swift):
```bash
cd v2
conda env create -f environment.yml
```
> **Note:** Flash attention installation can be tricky.

### Dataset
Download the raw datasets from [datasets.zip](https://www.dropbox.com/scl/fi/teo5pn8t296joue5c8pim/datasets.zip?rlkey=xvcgtdd9vlycep3nw3k17lfae&st=qd21069y&dl=0).

* Unzip `datasets.zip` to `./datasets`
* Unzip `datasets/nyc/raw.zip` to `datasets/nyc`
* Unzip `datasets/tky/raw.zip` to `datasets/tky`
* Unzip `datasets/ca/raw.zip` to `datasets/ca`
* For the CA dataset, run `python preprocessing/generate_ca_raw.py`

### Preprocess
We observe that you can achieve good performance simply by using user history alone, without trajectory similarity.

```bash
cd preprocessing
python run.py -f conf/best_conf/{dataset_name}.yml

cd ../v2
python convert_prompt_llm4poi.py \
    --dataset {dataset_name} \
    --train_csv ../datasets/{dataset_name}/preprocessed/train_sample.csv \
    --test_csv ../datasets/{dataset_name}/preprocessed/test_sample_with_traj.csv \
    --out_dir ../datasets/{dataset_name}/llm4poi_v2 \
    --history_limit 50
```

### Main Performance
#### Train
```bash
cd v2
MODEL_PATH=/path/to/base-model \
MODEL_TYPE=qwen2_5 \
DATASET_PATH=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/output \
bash sft.sh
```

#### Test
```bash
cd v2
MODEL_DIR=/path/to/base-model \
ENABLE_LORA=1 \
ADAPTER_DIR=/path/to/output \
SERVE_NAME=sft-lora \
bash serve_vllm.sh

DATASET_PATH=/path/to/test.jsonl \
OUTPUT_PATH=/path/to/predictions.jsonl \
MODEL_NAME=sft-lora \
bash eval.sh
```

</details>

---

<details>
<summary>v1: Legacy</summary>
<br>

> **Note:** Original implementation for the SIGIR 2024 paper.

### Install
1. Clone this repository to your local machine.
2. Install the environment by running
```bash
conda env create -f environment.yml
```
Alternatively, you can download the conda environment in Linux directly with this [google drive link](https://drive.google.com/file/d/1SKKSwjdEapQh5WOEpv8XkLZTTkhlKDg6/view?usp=sharing).
Then try:

```bash
mkdir -p llm4poi
tar -xzf "venv.tar.gz" -C "llm4poi"
conda activate llm4poi
```

3. Download the model from [Yukang/Llama-2-7b-longlora-32k-ft](https://huggingface.co/Yukang/Llama-2-7b-longlora-32k-ft)

### Dataset
Download the raw datasets from [datasets.zip](https://www.dropbox.com/scl/fi/teo5pn8t296joue5c8pim/datasets.zip?rlkey=xvcgtdd9vlycep3nw3k17lfae&st=qd21069y&dl=0).

* Unzip `datasets.zip` to `./datasets`
* Unzip `datasets/nyc/raw.zip` to `datasets/nyc`
* Unzip `datasets/tky/raw.zip` to `datasets/tky`
* Unzip `datasets/ca/raw.zip` to `datasets/ca`
* For the CA dataset, run `python preprocessing/generate_ca_raw.py`

### Preprocess
```bash
cd preprocessing
python run.py -f conf/best_conf/{dataset_name}.yml
python traj_qk.py -dataset_name {dataset_name}
cd ..

python traj_sim.py --dataset_name {dataset_name} --model_path {your_model_path}
python preprocessing/to_nextpoi_kqt.py -dataset_name {dataset_name}
```

### Main Performance
#### Train
```bash
torchrun --nproc_per_node=8 supervised-fine-tune-qlora.py \
  --model_name_or_path {your_model_path} \
  --bf16 True \
  --output_dir {your_output_path} \
  --model_max_length 32768 \
  --use_flash_attn True \
  --data_path datasets/processed/{DATASET_NAME}/train_qa_pairs_kqt.json \
  --low_rank_training True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type "constant_with_warmup" \
  --logging_steps 1 \
  --deepspeed "ds_configs/stage2.json" \
  --tf32 True
```

#### Test
```bash
python eval_next_poi.py --model_path {your_model_path} --dataset_name {DATASET_NAME} --output_dir {your_finetuned_model} --test_file "test_qa_pairs_kqt.txt"
```

</details>

---

## Acknowledgement
This code is developed based on [STHGCN](https://github.com/ant-research/Spatio-Temporal-Hypergraph-Model) and [LongLoRA](https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file).

## Citation
If you find our work useful, please consider cite our paper with following:
```bibtex
@inproceedings{li-2024-large,
author = {Li, Peibo and de Rijke, Maarten and Xue, Hao and Ao, Shuang and Song, Yang and Salim, Flora D.},
booktitle = {SIGIR 2024: 47th international ACM SIGIR Conference on Research and Development in Information Retrieval},
date-added = {2024-03-26 23:47:40 +0000},
date-modified = {2024-03-26 23:48:47 +0000},
month = {July},
publisher = {ACM},
title = {Large Language Models for Next Point-of-Interest Recommendation},
year = {2024}}
```
