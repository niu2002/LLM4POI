# LLM4POI v2 使用说明

这份说明只覆盖仓库里推荐的 `v2` 流程。

## 1. v2 依赖什么

`v2` 不依赖缺失的 `gptneox_attn_replace.py`，所以不会被那个旧版缺文件直接卡住。

你仍然需要自己准备：
- 数据集原始压缩包
- 预处理后的 CSV
- 一个可用于 `ms-swift` 的基础模型
- 如果做推理评测，还需要训练输出目录或 LoRA adapter

## 2. 数据集下载地址

README 里给出的数据集地址是：
- [datasets.zip](https://www.dropbox.com/scl/fi/teo5pn8t296joue5c8pim/datasets.zip?rlkey=xvcgtdd9vlycep3nw3k17lfae&st=qd21069y&dl=0)

解压后建议目录结构如下：

```text
datasets/
  nyc/
    raw/
  tky/
    raw/
  ca/
    raw/
```

对于 `ca` 数据集，还需要执行：

```bash
python preprocessing/generate_ca_raw.py
```

## 3. 模型是什么

`v2` 没有把模型固定成 README 里的某一个 Hugging Face 名称，而是让你通过环境变量传入：
- `MODEL_PATH`：基础模型路径或模型 id
- `MODEL_TYPE`：`ms-swift` 识别的模型类型

例如脚本里会这样传：

```bash
MODEL_PATH=/path/to/base-model
MODEL_TYPE=qwen2_5
```

具体 `MODEL_TYPE` 要和你实际选的模型匹配。

## 4. 预处理流程

先进入预处理目录：

```bash
cd preprocessing
python run.py -f conf/best_conf/{dataset_name}.yml
```

然后回到 `v2`，把 CSV 转成训练/评测数据：

```bash
cd ../v2
python convert_prompt_llm4poi.py \
  --dataset {dataset_name} \
  --train_csv ../datasets/{dataset_name}/preprocessed/train_sample.csv \
  --test_csv ../datasets/{dataset_name}/preprocessed/test_sample_with_traj.csv \
  --out_dir ../datasets/{dataset_name}/llm4poi_v2 \
  --history_limit 50
```

`out_dir` 下通常会得到适合 `v2` 训练和评测使用的数据文件。

## 5. 训练

已修好的脚本是：
- [v2/sft.sh](/F:/jjy/LLM4POI/v2/sft.sh)

最小用法：

```bash
cd v2
MODEL_PATH=/path/to/base-model \
MODEL_TYPE=qwen2_5 \
DATASET_PATH=/path/to/train.jsonl \
OUTPUT_DIR=/path/to/output \
bash sft.sh
```

必须提供的环境变量：
- `MODEL_PATH`
- `MODEL_TYPE`
- `DATASET_PATH`
- `OUTPUT_DIR`

## 6. 启动 vLLM 服务

已修好的脚本是：
- [v2/serve_vllm.sh](/F:/jjy/LLM4POI/v2/serve_vllm.sh)

基础模型直推：

```bash
cd v2
MODEL_DIR=/path/to/base-model \
bash serve_vllm.sh
```

带 LoRA adapter：

```bash
cd v2
MODEL_DIR=/path/to/base-model \
ENABLE_LORA=1 \
ADAPTER_DIR=/path/to/output \
SERVE_NAME=sft-lora \
bash serve_vllm.sh
```

## 7. 评测

已修好的脚本是：
- [v2/eval.sh](/F:/jjy/LLM4POI/v2/eval.sh)

最小用法：

```bash
cd v2
DATASET_PATH=/path/to/test.jsonl \
OUTPUT_PATH=/path/to/predictions.jsonl \
MODEL_NAME=sft-lora \
bash eval.sh
```

常用可选变量：
- `BASE_URL`
- `API_KEY`
- `SYSTEM_PROMPT`
- `MAX_NEW_TOKENS`
- `CONCURRENCY`
- `TEMPERATURE`

## 8. 现在 v2 还缺什么

我已经修掉了：
- README 里 `v2` 的错误命令名
- `sft.sh` 的参数拼接问题
- `serve_vllm.sh` 的语法错误
- `eval.sh` 的语法错误

但你现在如果要真正跑通，仍然需要你自己准备：
- 外部数据集
- 实际模型
- 可用的 GPU 环境
- 与模型匹配的 `MODEL_TYPE`

## 9. 结论

`v2` 可以修，而且已经比原仓库更接近“能直接执行”的状态。

它现在最大的阻碍已经不是脚本缺失，而是外部资源是否齐全：
- 数据集有没有下好
- 模型有没有准备好
- 环境依赖有没有装齐
