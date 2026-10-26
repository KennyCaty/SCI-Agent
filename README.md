# SCI

## 1. Install

```bash
cd SCI_code
conda create -n sci python=3.11 -y
conda activate sci
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
npm ci
npm run build
python - <<'PY'
import tiktoken
tiktoken.get_encoding('o200k_base')
PY
sci download --models
```

Models are downloaded to `models/Qwen3-Embedding-0.6B/` and `models/GLiNER2.5-small/` and shared by both datasets.

## 2. Configure the model API

Copy the API configuration template:

```bash
cp .env.example .env
```

Edit `.env` to set separate API bases and keys for DeepSeek-V4.1-Flash, Qwen3.8-27B, Qwen3.8-Flash, GPT-6 Luna, and the judge, and set `SCI_JUDGE_MODEL` to the judge model name. Fill in only the models you will use. Each group is independent. API bases should not include the final `/responses` or `/chat/completions` path.

Choose the corresponding configuration under `configs/models/` and set `model` and `api` to match your endpoint.

| Configuration file | API model name `model` |
| --- | --- |
| `deepseek-v4.1-flash.json` | `deepseek-flash` |
| `qwen3.8-27b.json` | `Qwen3.8-27B` |
| `qwen3.8-flash.json` | `qwen3.8-flash` |
| `gpt-6-luna.json` | `gpt-6-luna` |

Set prices in `pricing.tiers` in the same file, in USD per million tokens. `input`, `output`, `cache_read`, and `cache_write` cover uncached input, output, cache reads, and cache writes. Defaults use DeepSeek off-peak, Qwen Flash international, and Luna Standard reference rates. Adjust these rates when changing providers. Qwen3.8-27B defaults to self-hosted operation and records tokens only. For a paid endpoint, replace its `pricing` with a configuration containing `currency` and `tiers`. Per-question costs are saved in `TOKEN_USAGE.json`.

Then load `.env` in the terminal used to run the commands:

```bash
set -a
source .env
set +a
```

## 3. BrowseComp-Plus (BCP)

Download the corpus and questions to `data/raw/corpus.jsonl` and `data/raw/questions.jsonl`, then build the index. Run the build steps sequentially. Proceed to inference when `sci index status` reports `READY`.

```bash
sci download --corpus --questions
export SCI_INDEX_CONFIG=configs/indexing/browsecomp_plus.json
sci index prepare data/raw/corpus.jsonl
sci index ner
sci index inputs
sci index build
sci index status
unset SCI_INDEX_CONFIG
```

Run inference. Replace the model configuration path to use another model.

```bash
sci batch configs/models/deepseek-v4.1-flash.json \
  data/raw/questions.jsonl outputs/bcp-run1
```

Evaluate with the judge API and model configured in step 2. The judge API must support Responses.

```bash
sci evaluate data/raw/questions.jsonl \
  outputs/bcp-run1/predictions.jsonl outputs/bcp-run1-grades \
  --model "$SCI_JUDGE_MODEL"
```

Predictions: `outputs/bcp-run1/predictions.jsonl`. Evaluation: `outputs/bcp-run1-grades/summary.json`.

## 4. TREC-COVID

Download the dataset to `data/beir/trec-covid/`, then build its index. Proceed when the status is `READY`.

```bash
sci download --trec-covid
export SCI_INDEX_CONFIG=configs/indexing/trec_covid.json
sci index prepare --trec-covid data/beir/trec-covid/corpus.jsonl
sci index ner
sci index inputs
sci index build
sci index status
unset SCI_INDEX_CONFIG
```

Set `provider_config` in `configs/benchmarks/trec_covid.json` to the model configuration to use (default: `../models/gpt-6-luna.json`). Prepare the run, perform inference, and evaluate:

```bash
sci trec-covid prepare configs/benchmarks/trec_covid.json \
  data/beir/trec-covid outputs/trec-run1
sci trec-covid run outputs/trec-run1 --concurrency 1
sci trec-covid evaluate outputs/trec-run1
```

Evaluation: `outputs/trec-run1/TREC_COVID_EVALUATION.json`. No judge API is needed.

Use a new output directory name for each run and evaluation. Build BCP and TREC-COVID sequentially. Build logs are under `outputs/indexing/logs/`.
