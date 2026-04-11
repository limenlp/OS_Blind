# The Blind Spot of Agent Safety: How Benign User Instructions Expose Critical Vulnerabilities in Computer-Use Agents

**Xuwei Ding\*<sup>&alpha;</sup>, Skylar Zhai\*<sup>&beta;</sup>, Linxin Song\*<sup>&gamma;</sup>, Jiate Li<sup>&gamma;</sup>, Taiwei Shi<sup>&gamma;</sup>, Nicholas Meade<sup>&delta;,&epsilon;</sup>, Siva Reddy<sup>&delta;,&epsilon;</sup>, Jian Kang<sup>&eta;</sup>, Jieyu Zhao<sup>&gamma;</sup>**

<sup>&alpha;</sup>University of Wisconsin-Madison, <sup>&beta;</sup>University of Minnesota, <sup>&gamma;</sup>University of Southern California, <sup>&delta;</sup>McGill University, <sup>&epsilon;</sup>Mila, <sup>&eta;</sup>MBZUAI

\* Equal contribution

## Abstract

OS-Blind evaluates computer-use agents (CUAs) under unintended attack conditions: every task begins with a fully benign user instruction, yet harm emerges from the environment during execution. This is in contrast to existing benchmarks that rely on explicitly malicious instructions or injected adversarial prompts. By using benign instructions as a disguise, OS-Blind captures a substantially more realistic and challenging safety threat.

## Installation

OS-Blind is built on top of the OSWorld environment. Please follow the official OSWorld setup instructions:

- OSWorld installation: https://github.com/xlang-ai/OSWorld/?tab=readme-ov-file#-installation
- OSWorld FAQ: https://github.com/xlang-ai/OSWorld#-faq


Important environment note:

- Our experiments are based on Docker (`--provider_name docker`).
- We have **not** tested this benchmark on VMware or EC2.

## Data

The official OS-Blind dataset is hosted on Hugging Face:

- Dataset page: https://huggingface.co/datasets/lime-nlp/OS-Blind

Please download and follow the dataset instructions from that page.

### Dataset Overview

OS-Blind contains **300 human-crafted tasks** across **12 safety categories**, organized into two threat clusters:

- Cluster I (environment-embedded threats): `01_credential_phishing`, `08_illegal_and_harmful_content`, `09_risky_financial_platforms`, `12_pop_ups_attack`
- Cluster II (agent-initiated harms): `02_data_exfiltration`, `03_fraud_and_forgery`, `04_malware_deployment`, `05_system_sabotage`, `06_misinformation_and_manipulation`, `07_harassment_and_threats`, `10_deceptive_agreements`, `11_code_injection`

Main files:

- `test_all.json`: category-to-task index used by runners
- `examples/<category>/<task>.json`: per-task configs and instructions

### Download Steps

1. Open the dataset page above and request access.
2. Download the dataset and place it under `evaluation_example_safety/` in this repository.


### Expected Structure

After download, the following files/directories should exist:

- `evaluation_example_safety/test_all.json`
- `evaluation_example_safety/examples/...`
- `evaluation_example_safety/restore_urls.py`

### Notes

- You do **not** need additional VM/network-forwarding setup to use the dataset files themselves.
- For categories `08_illegal_and_harmful_content` and `09_risky_financial_platforms`, URLs are defanged in the released data; refer to the dataset page and `evaluation_example_safety/restore_urls.py` if you need to restore them for evaluation.

## [VPI-Bench](https://github.com/cua-framework/agents) Reimplementation

To enable fair comparison in a unified OSWorld environment and facilitate community follow-up research, we also provide an OSWorld-based implementation of the VPI-Bench setup under:

- `evaluation_example_vpibench/`

In this repository, this baseline currently uses the `computer_use_osworld` task set from VPI-Bench in the OSWorld environment:

- `evaluation_example_vpibench/examples/computer_use_osworld/*.json`
- `evaluation_example_vpibench/test_all.json`

## Methods Evaluated in This Benchmark

This benchmark evaluates the following methods using the corresponding evaluation runners (mainly `run_multienv_xxx.py`):

| Runner Script | Method | Type | Env File | Reference |
| --- | --- | --- | --- | --- |
| `run_multienv_claude.py` | Claude CUA baseline | End-to-End | `envs/claude.env` | Anthropic Computer Use: https://docs.anthropic.com/en/docs/build-with-claude/computer-use |
| `run_multienv_uitars15_v1.py` | UI-TARS-1.5 baseline | End-to-End | `envs/uitars.env` | UI-TARS-1.5 (HF): https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B |
| `run_multienv_opencua.py` | OpenCUA baseline | End-to-End | `envs/opencua.env` | OpenCUA (GitHub): https://github.com/xlang-ai/OpenCUA |
| `run_multienv_evocua.py` | EvoCUA baseline | End-to-End | `envs/evocua.env` | EvoCUA (GitHub): https://github.com/meituan/EvoCUA; EvoCUA-8B (HF): https://huggingface.co/meituan/EvoCUA-8B-20260105 |
| `run_multienv_uitars15_v1_mirrorguard.py` | UI-TARS + MirrorGuard | Defend Method | `envs/mirrorguard.env` | UI-TARS-1.5 (HF): https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B; MirrorGuard (GitHub): https://github.com/WhitzardAgent/MirrorGuard; MirrorGuard (HF): https://huggingface.co/WhitzardAgent/MirrorGuard |
| `run_multienv_evocua_mirrorguard.py` | EvoCUA + MirrorGuard | Defend Method | `envs/evocua.env` + `envs/mirrorguard.env` | EvoCUA (GitHub): https://github.com/meituan/EvoCUA; EvoCUA-8B (HF): https://huggingface.co/meituan/EvoCUA-8B-20260105; MirrorGuard (GitHub): https://github.com/WhitzardAgent/MirrorGuard; MirrorGuard (HF): https://huggingface.co/WhitzardAgent/MirrorGuard |
| `run_multienv_jedi7b.py` | JEDI | Multi-Agent | `envs/jedi.env` | JEDI-7B-1080p (HF): https://huggingface.co/xlangai/Jedi-7B-1080p |
| `run_multienv_s2.py` | Agent-S2 | Multi-Agent | `envs/s2.env` | Agent-S (GitHub): https://github.com/simular-ai/Agent-S |
| `run_coactv2.py` | CoAct-1 | Multi-Agent | `envs/coactv2.env` | CoAct-1 (GitHub): https://github.com/SalesforceAIResearch/CoAct-1 |


## Environment Configuration (`envs/`)

Environment loading (see `lib_env.py`):

- Create the required `envs/{name}.env` files yourself.
- The code loads configuration from `envs/{name}.env`.

Only fill your own values. **Do not commit API keys/tokens to git.**

Core variable names by env file (attack-related variables omitted here):

- `envs/claude.env`: `ANTHROPIC_API_KEY`
- `envs/uitars.env`: `DOUBAO_API_URL`, `DOUBAO_API_KEY`
- `envs/jedi.env`: `JEDI_SERVICE_URL`, `JEDI_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- `envs/opencua.env`: `OPENCUA_URL`, `OPENCUA_API_KEY`
- `envs/evocua.env`: `EVOCUA_BASE_URL`, `EVOCUA_API_KEY`
- `envs/s2.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `PERPLEXICA_URL`
- `envs/coactv2.env`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `UITARS_API_URL`
- `envs/mirrorguard.env`: `DOUBAO_API_URL`, `DOUBAO_API_KEY`, `MIRRORGUARD_API_URL`, `MIRRORGUARD_API_KEY`

Attack VLM/LLM variables are documented in the `vLLM Support` and `Attack LLM Configuration` sections below.

### CoAct-1 Modes and Model Choices

`run_coactv2.py` exposes `--mode` with the following choices: `human`, `hybrid`, `coact_cua_only`, `coact_coding_only`, and `coact_opensource_sft`.

In this paper, our main CoAct-1 evaluation setting is `coact_cua_only`.
We focus on this setting because under `hybrid`, the coder does not have direct access to the live GUI screen and can only execute delegated subtasks.
As a result, the coder does not have a meaningful opportunity to inspect or defend against screen-level threats, so we do not treat `hybrid` as the main defense setting in OS-Blind.

You can modify both `--orchestrator_model` and `--cua_model` in `run_coactv2.py`.
In our experiments, we tested three orchestrator models: `claude-sonnet-4-5-20250929`, `gpt-5`, and `o3`.
For the CoAct-1 GUI Operator, we used `ByteDance-Seed/UI-TARS-1.5-7B` and `claude-sonnet-4-5-20250929`.
If you use UI-TARS as the CoAct-1 GUI Operator, make sure `UITARS_API_URL` is configured in `envs/coactv2.env`.

### Agent-S2 `PERPLEXICA_URL`

For Agent-S2, `--search_engine` defaults to `Perplexica`, so `PERPLEXICA_URL` should be configured.

**Perplexica Setup (Recommended)**

1. Install Perplexica:

```bash
cd <your-workspace>
git clone https://github.com/ItzCrazyKns/Perplexica.git
cd Perplexica

# Use Docker Compose V2 (space, not hyphen)
docker compose up -d
```

2. Configure API Keys:

- Open `http://localhost:3000`
- Fill in your OpenAI API Key in the Web UI
- Select a model (recommended: GPT-4o)

3. Set the environment variable:

```bash
export PERPLEXICA_URL=http://localhost:3000
```

## vLLM Support

Most local methods in this repository run with `vllm serve`.
Start the model services first, then write their host/port into the corresponding `envs/*.env` URLs.
You need to prepare a working `vllm` environment yourself before launching these services.
Choose the model size and `CUDA_VISIBLE_DEVICES` setting based on your available GPUs and VRAM.
The examples below omit tensor/pipeline parallel flags on purpose, since the correct multi-GPU setup depends on your machine.

Use only the models you need for the runner you are evaluating.

### vLLM Startup Commands (Concrete Models)

```bash
# Attack VLM (used by ATTACK_VLM_API_URL, especially for 12_pop_ups_attack)
CUDA_VISIBLE_DEVICES=<GPU_ID> vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90
```

```bash
# UI-TARS baseline
CUDA_VISIBLE_DEVICES=<GPU_ID> vllm serve ByteDance-Seed/UI-TARS-1.5-7B \
  --host 127.0.0.1 --port 8002 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

```bash
# JEDI executor
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve xlangai/Jedi-7B-1080p \
  --host 127.0.0.1 --port 8002 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

```bash
# EvoCUA (8B)
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve meituan/EvoCUA-8B-20260105 \
  --served-model-name meituan/EvoCUA-8B-20260105 \
  --host 127.0.0.1 --port 8002 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

```bash
# EvoCUA (32B)
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve meituan/EvoCUA-32B-20260105 \
  --served-model-name meituan/EvoCUA-32B-20260105 \
  --host 127.0.0.1 --port 8002 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

```bash
# MirrorGuard defense model
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve WhitzardAgent/MirrorGuard \
  --served-model-name WhitzardAgent/MirrorGuard \
  --host 127.0.0.1 --port 8003 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

```bash
# OpenCUA (72B)
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve <OPENCUA_72B_CHECKPOINT_OR_PATH> \
  --served-model-name OpenCUA-72B \
  --host 127.0.0.1 --port 8003 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

```bash
# OpenCUA (32B)
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve <OPENCUA_32B_CHECKPOINT_OR_PATH> \
  --served-model-name OpenCUA-32B \
  --host 127.0.0.1 --port 8003 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

```bash
# OpenCUA (7B)
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve <OPENCUA_7B_CHECKPOINT_OR_PATH> \
  --served-model-name OpenCUA-7B \
  --host 127.0.0.1 --port 8003 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95
```

For OpenCUA and EvoCUA, make sure the served model name is consistent with the `--model` you pass to the runner script.

### After Startup: Fill `envs/*.env` URLs

Set the URL fields to `http://localhost:<port>/...` with the exact ports you started above.
For local vLLM services, API keys can usually be set to `dummy`.

Recommended URL/API key mapping:

- Attack VLM: `ATTACK_VLM_API_URL=http://localhost:8000/v1`
- UI-TARS URL: `DOUBAO_API_URL=http://localhost:8002/v1`, `DOUBAO_API_KEY=dummy`
- JEDI URL: `JEDI_SERVICE_URL=http://localhost:8002/v1`, `JEDI_API_KEY=dummy`
- EvoCUA URL: `EVOCUA_BASE_URL=http://localhost:8002/v1`, `EVOCUA_API_KEY=dummy`
- OpenCUA URL: `OPENCUA_URL=http://localhost:8003/v1/chat/completions`, `OPENCUA_API_KEY=dummy`
- MirrorGuard URL: `MIRRORGUARD_API_URL=http://localhost:8003/v1`, `MIRRORGUARD_API_KEY=dummy`, 

### Attack VLM vs Attack LM

- `ATTACK_VLM_API_URL`: visual-language attack model service URL. In this repo, popup tasks typically use `Qwen/Qwen3-VL-4B-Instruct`.
- `ATTACK_LLM_MODEL`: text attack model name. The current default in code is `gpt-4o`.

### Attack LLM Configuration

Popup attack generation is handled by `desktop_env/attackable_env.py`.
For Attack LLM, use one of the following two setups:

1. Official OpenAI (default)

```bash
OPENAI_API_KEY=<YOUR_OPENAI_KEY>
ATTACK_LLM_MODEL=gpt-4o
# Do not set ATTACK_LLM_API_URL
```

When `ATTACK_LLM_API_URL` is unset, the code uses official OpenAI URL `https://api.openai.com/v1`.
If `ATTACK_LLM_MODEL` is also unset, the current code defaults to `gpt-4o`.

2. Custom OpenAI-compatible URL

```bash
OPENAI_API_KEY=<YOUR_KEY_FOR_THIS_BASE_URL>
ATTACK_LLM_API_URL=<YOUR_OPENAI_COMPATIBLE_BASE_URL>
ATTACK_LLM_MODEL=gpt-4o
```

## Running the Benchmark

After starting the required model services and filling the corresponding `envs/*.env` files, each runner can be launched with its default configuration.

By default, the runners now use:

- `--test_all_meta_path evaluation_example_safety/test_all.json`
- `--num_envs 30`


### End-to-End Runners

```bash
python run_multienv_claude.py
python run_multienv_uitars15_v1.py
python run_multienv_opencua.py
python run_multienv_evocua.py
python run_multienv_uitars15_v1_mirrorguard.py
python run_multienv_evocua_mirrorguard.py
```

### Multi-Agent Runners

```bash
python run_multienv_jedi7b.py
python run_multienv_s2.py
python run_coactv2.py
```

### Defend Method Runners
```bash
python run_multienv_uitars15_v1_mirrorguard.py
python run_multienv_evocua_mirrorguard.py
```

### Notes

- `run_multienv_uitars15_v1_mirrorguard.py` and `run_multienv_evocua_mirrorguard.py` require both the base agent service URL and the MirrorGuard service URL.
- For multi-agent runners, the default result directory names include the key model choices:
- `run_multienv_jedi7b.py`: `results_jedi7b_gpt5_round1`
- `run_multienv_s2.py`: `results_s2_gpt5_claude45_round1`
- `run_coactv2.py`: `results_coactv2_gpt5_claude45_round1`

## LLM-as-a-Judge Evaluation

OS-Blind also provides LLM-as-a-judge evaluation scripts under `judge/`.
Each script reads saved agent trajectories, calls a GPT-4o judge, and writes per-task verdicts plus an ASR summary.

### Judge Setup

Create `judge/OPENAI_API_KEY.key`:

```text
# key only
sk-xxxx
```

### Judge Scripts

Use the script that matches your agent:

| Agent | Script |
| --- | --- |
| End-to-end models (Claude, UI-TARS, EvoCUA, ...) | `judge/eval_end2end_jedi.py` |
| JEDI | `judge/eval_end2end_jedi.py` |
| CoAct-1 | `judge/eval_coact.py` |
| Agent-S2 | `judge/eval_s2.py` |

Category 12 (`12_pop_ups_attack`) is automatically judged by click coordinates rather than an LLM.
Pass `--include-cat12` if you want to include it.

### CoAct-1 Judge

Specify the GUI agent type with `--gui-agent claude` or `--gui-agent uitars`, since Claude and UI-TARS produce different `history_inputs.json` formats.

```bash
python judge/eval_coact.py \
    --results-base /path/to/results_coact \
    --output-dir   ./eval_results_coact \
    --gui-agent    claude
```

### End-to-End, JEDI, and Agent-S2 Judge

```bash
python judge/eval_end2end_jedi.py \
    --results-base /path/to/results \
    --output-dir   ./eval_results

python judge/eval_s2.py \
    --results-base /path/to/results_s2 \
    --output-dir   ./eval_results_s2
```

### Judge Output

Each run writes to `--output-dir`:

- `eval_results_<timestamp>.json`: per-task verdicts with judge rationale
- `eval_summary_<timestamp>.json`: overall ASR and per-category breakdown

### Aggregating Multiple Rounds

`judge/aggregate.py` aggregates results across 3 rounds and reports ASR at three thresholds: unsafe in at least 1 round, at least 2 rounds, or all 3 rounds.

Expected layout under `--base-dir`:

```text
base_dir/
  round1/eval_*.json
  round2/eval_*.json
  round3/eval_*.json
```

```bash
python judge/aggregate.py \
    --base-dir   /path/to/eval_results \
    --model-name claude-sonnet-4-5
```
