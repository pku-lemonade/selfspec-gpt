# gpt-fast
Simple and efficient pytorch-native transformer text generation.

Featuring:
1. Very low latency
2. <1000 lines of python
3. No dependencies other than PyTorch and sentencepiece
4. int8/int4 quantization
5. Speculative decoding
6. Tensor parallelism
7. Supports Nvidia and AMD GPUs

This is *NOT* intended to be a "framework" or "library" - it is intended to show off what kind of performance you can get with native PyTorch :) Please copy-paste and fork as you desire.

For an in-depth walkthrough of what's in this codebase, see this [blog post](https://pytorch.org/blog/accelerating-generative-ai-2/).

## Paper Status

This repository is being used as the functional simulator for the roadmap in
[`references/Roadmap.md`](./references/Roadmap.md).

If the goal is to write the paper described by that roadmap, the required
result categories are:

1. Functional speculative-decoding accuracy / acceptance results.
2. ADC / DAC interface-precision sweeps.
3. Noise-tolerance and fine-tuning impact results.
4. Layer-sensitivity / draft-allocation results.
5. Hardware-estimator outputs:
   tokens/s, latency, energy, area, and break-even prompt length.

### What Is Already Done

The following categories are already available in this repository:

- Roadmap-style acceptance summaries for the current best checkpoints.
- `k` sweeps for the main roadmap models.
- ADC-bit sweeps for the older absolute-readout path.
- Predictive-delta ablations for:
  - verify-side delta readout
  - verify DAC sensitivity
  - draft-side delta readout
  - draft DAC sensitivity
- A full `k x draft_adc_bits` matrix for:
  - verify delta readout
  - draft delta readout
  - `verify_adc_bits = 8`
  - `verify_delta_dac_bits = 8`
  - `draft_delta_dac_bits = 8`
- Clean-target quality screens for the three main paper models.

### What Is Still Missing In This Repo

For the current paper scope used in this repository:

- the functional-simulator side is ready enough for paper writing
- the main paper result files are already on disk

The only remaining simulator-repo work should be considered cleanup / polish:

- keep the OpenSpec docs aligned with the code as delta-readout support evolves
- optionally add more summary files or figure-generation helpers for paper plots

If the hardware estimator is maintained elsewhere, then there is no major
simulator-side blocker left in this repository for drafting the paper.

## Paper Files

If writing the paper now, these are the primary result files to use.

### 1. High-Level Roadmap Winners

- [`out/final_roadmap_model_summary.md`](./out/final_roadmap_model_summary.md)

This is the best starting point for the narrative section that answers:

- which checkpoint is the current winner for each model size
- what the roadmap-style large-slice acceptance is
- how the tuned checkpoint compares with base

Supporting large-slice roadmap acceptance files:

- `Qwen3-0.6B`
  - [`out/corrected_qwen0p6b_roadmap_awpqkvout_step50_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json`](./out/corrected_qwen0p6b_roadmap_awpqkvout_step50_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json)
  - [`out/corrected_qwen0p6b_roadmap_base_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json`](./out/corrected_qwen0p6b_roadmap_base_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json)
- `Llama-3.2-1B`
  - [`out/corrected_llama3p2_1b_oldtuned_targetself_write10pct_rel_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json`](./out/corrected_llama3p2_1b_oldtuned_targetself_write10pct_rel_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json)
  - [`out/corrected_llama3p2_1b_base_write10pct_rel_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json`](./out/corrected_llama3p2_1b_base_write10pct_rel_s500_flex_cuda1_20260316/stats_meta_Lprompt_64.json)
- `Qwen3-1.7B`
  - [`out/corrected_qwen3_1p7b_oldtuned_targetself_write10pct_rel_s500_flex_cuda1_20260317/stats_meta_Lprompt_64.json`](./out/corrected_qwen3_1p7b_oldtuned_targetself_write10pct_rel_s500_flex_cuda1_20260317/stats_meta_Lprompt_64.json)

### 2. Clean-Target Quality Screens

Use these for the “the clean target has not obviously collapsed” table:

- [`out/qwen0p6b_quality_screen.json`](./out/qwen0p6b_quality_screen.json)
- [`out/llama3p2_1b_quality_screen.json`](./out/llama3p2_1b_quality_screen.json)
- [`out/qwen3_1p7b_quality_screen.json`](./out/qwen3_1p7b_quality_screen.json)

### 3. K-Sweep Result Summary

Use this for the “what burst length is best?” section of the paper:

- [`out/k_sweep_summary.md`](./out/k_sweep_summary.md)

Primary JSON files behind that summary:

- [`out/k_sweep_qwen0p6b_roadmap_s100_cuda1.json`](./out/k_sweep_qwen0p6b_roadmap_s100_cuda1.json)
- [`out/k_sweep_llama3p2_1b_roadmap_s100_cuda1.json`](./out/k_sweep_llama3p2_1b_roadmap_s100_cuda1.json)
- [`out/k_sweep_qwen3_1p7b_roadmap_s100_cuda1.json`](./out/k_sweep_qwen3_1p7b_roadmap_s100_cuda1.json)
- extended `k=9..12` files:
  - [`out/k_sweep_qwen0p6b_roadmap_s100_cuda1_k9to12.json`](./out/k_sweep_qwen0p6b_roadmap_s100_cuda1_k9to12.json)
  - [`out/k_sweep_llama3p2_1b_roadmap_s100_cuda1_k9to12.json`](./out/k_sweep_llama3p2_1b_roadmap_s100_cuda1_k9to12.json)
  - [`out/k_sweep_qwen3_1p7b_roadmap_s100_cuda1_k9to12.json`](./out/k_sweep_qwen3_1p7b_roadmap_s100_cuda1_k9to12.json)

### 4. Absolute ADC-Bit Sweep Summary

Use this for the old “no delta readout” ADC comparison baseline:

- [`out/k_sweep_adc_summary.md`](./out/k_sweep_adc_summary.md)

Representative supporting files:

- [`out/k_sweep_qwen0p6b_best_adc4_12_s100_cuda1.json`](./out/k_sweep_qwen0p6b_best_adc4_12_s100_cuda1.json)
- [`out/k_sweep_qwen0p6b_best_adc8_12_s100_cuda1.json`](./out/k_sweep_qwen0p6b_best_adc8_12_s100_cuda1.json)
- [`out/k_sweep_llama3p2_1b_best_adc4_12_s100_cuda1.json`](./out/k_sweep_llama3p2_1b_best_adc4_12_s100_cuda1.json)
- [`out/k_sweep_llama3p2_1b_best_adc8_12_s100_cuda1.json`](./out/k_sweep_llama3p2_1b_best_adc8_12_s100_cuda1.json)
- [`out/k_sweep_qwen3_1p7b_best_adc4_12_s100_cuda1.json`](./out/k_sweep_qwen3_1p7b_best_adc4_12_s100_cuda1.json)
- [`out/k_sweep_qwen3_1p7b_best_adc8_12_s100_cuda1.json`](./out/k_sweep_qwen3_1p7b_best_adc8_12_s100_cuda1.json)

### 5. Delta-Readout ADC Summary

Use this for the new “delta readout on both verify and draft, DAC fixed to 8-bit”
section:

- [`out/k_sweep_delta_adc_summary.md`](./out/k_sweep_delta_adc_summary.md)

Primary matrix JSONs for the paper:

- `Qwen3-0.6B`
  - [`out/k_sweep_qwen0p6b_delta_draftadc2_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen0p6b_delta_draftadc2_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen0p6b_delta_draftadc3_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen0p6b_delta_draftadc3_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen0p6b_delta_draftadc4_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen0p6b_delta_draftadc4_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen0p6b_delta_draftadc5_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen0p6b_delta_draftadc5_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen0p6b_delta_draftadc6_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen0p6b_delta_draftadc6_verifyadc8_deltadac8_draftdeltadac8_s100.json)
- `Llama-3.2-1B`
  - [`out/k_sweep_llama3p2_1b_delta_draftadc2_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_llama3p2_1b_delta_draftadc2_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_llama3p2_1b_delta_draftadc3_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_llama3p2_1b_delta_draftadc3_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_llama3p2_1b_delta_draftadc4_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_llama3p2_1b_delta_draftadc4_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_llama3p2_1b_delta_draftadc5_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_llama3p2_1b_delta_draftadc5_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_llama3p2_1b_delta_draftadc6_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_llama3p2_1b_delta_draftadc6_verifyadc8_deltadac8_draftdeltadac8_s100.json)
- `Qwen3-1.7B`
  - [`out/k_sweep_qwen3_1p7b_delta_draftadc2_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen3_1p7b_delta_draftadc2_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen3_1p7b_delta_draftadc3_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen3_1p7b_delta_draftadc3_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen3_1p7b_delta_draftadc4_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen3_1p7b_delta_draftadc4_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen3_1p7b_delta_draftadc5_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen3_1p7b_delta_draftadc5_verifyadc8_deltadac8_draftdeltadac8_s100.json)
  - [`out/k_sweep_qwen3_1p7b_delta_draftadc6_verifyadc8_deltadac8_draftdeltadac8_s100.json`](./out/k_sweep_qwen3_1p7b_delta_draftadc6_verifyadc8_deltadac8_draftdeltadac8_s100.json)

### 6. Predictive-Delta Ablation Files

Use these for the focused predictive-delta section on `Qwen3-1.7B`.

Verify delta readout ablations:

- baseline verify `12-bit` absolute:
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_absolute.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_absolute.json)
- verify `8-bit` delta, ideal DAC:
  - [`out/compare_qwen3_1p7b_adc4_k5_verify8_delta.json`](./out/compare_qwen3_1p7b_adc4_k5_verify8_delta.json)
- verify `8-bit` delta, DAC sweep:
  - [`out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac2.json`](./out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac2.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac4.json`](./out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac4.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac5.json`](./out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac5.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac6.json`](./out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac6.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac7.json`](./out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac7.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac8.json`](./out/compare_qwen3_1p7b_adc4_k5_verify8_delta_dac8.json)

Draft delta readout ablations:

- draft `4-bit` baseline, no draft delta:
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_absolute.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_absolute.json)
- draft `4-bit` with ideal draft delta:
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta.json)
- draft `4-bit` with draft DAC sweep:
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac2.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac2.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac4.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac4.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac5.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac5.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac6.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac6.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac7.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac7.json)
  - [`out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac8.json`](./out/compare_qwen3_1p7b_adc4_k5_verify12_draftdelta_dac8.json)

## Supported Models

### LLaMA family
Please check the rest of this page about benchmark of LLaMA family models.

### Mixtral 8x7B
We also supported [Mixtral 8x7B](https://mistral.ai/news/mixtral-of-experts/) which is a high-quality sparse mixture of experts (MoE) model, the average token generation rates are:

|                  |   1 GPU |    2 GPU  | 4 GPU  |    8 GPU   |
|------------------|---------|-----------|--------|------------|
|baseline(bfloat16)|    OOM  |    96.67  | 155.35 |  227.82    |
|        int8      |   97.92 |   155.03  | 216.87 |  279.35    |

Note that the benchmarks run on an 8xA100-80GB, power limited to 330W with a hybrid cube mesh topology. Note that all benchmarks are run at *batch size=1*, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a very small prompt length (just 5 tokens).

For more details about Mixtral 8x7B, please check [this page](./mixtral-moe) or this [note](https://thonking.substack.com/p/short-supporting-mixtral-in-gpt-fast).

## Examples
In the spirit of keeping the repo minimal, here are various examples of extensions you can make to gpt-fast as PRs.
- [Google Gemma](https://github.com/meta-pytorch/gpt-fast/pull/115)
- [xAI Grok-1](https://github.com/meta-pytorch/gpt-fast/pull/171)
- [Databricks DBRX](https://github.com/meta-pytorch/gpt-fast/pull/174)

## Community

Projects inspired by gpt-fast in the community:

- [gpt-blazing](https://github.com/armed-gpt/gpt-blazing): applies the same performance optimization strategy to more models (e.g., baichuan2).
- [gptfast](https://github.com/MDK8888/GPTFast): applies a subset of the performance optimizations to all Huggingface models
- [gpt-accelera](https://github.com/Edward-Sun/gpt-accelera): extends `gpt-fast` to SFT/RM/PPO training and batched inference to optimize the throughput

## Installation
[Download PyTorch nightly](https://pytorch.org/get-started/locally/)

Install required packages:

```bash
pip install -r requirements.txt
```

To download llama models, go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access.
Then login with `huggingface-cli login`



## Downloading Weights
Models tested/supported
```text
tinyllamas/stories{15,42,100}
openlm-research/open_llama_7b
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf
codellama/CodeLlama-7b-Python-hf
codellama/CodeLlama-34b-Python-hf
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.1
mistralai/Mistral-7B-Instruct-v0.2
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3.1-8B
meta-llama/Llama-3.2-1B
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Meta-Llama-3.1-70B
meta-llama/Meta-Llama-3.1-405B
```

For example, to convert Llama-2-7b-chat-hf
```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_REPO
```

To run a ~1B Llama-family model end-to-end, you can use the included flow script:
```bash
export HF_TOKEN=...  # required for gated Meta repos
./scripts/test_flow_llama_small.sh meta-llama/Llama-3.2-1B-Instruct
```

## Benchmarks
Benchmarks run on an 8xA100-80GB, power limited to 330W with a hybrid cube mesh topology. Note that all benchmarks are run at *batch size=1*, making the reported tokens/s numbers equivalent to "tokens/s/user". In addition, they are run with a very small prompt length (just 5 tokens).

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  104.9  | 1397.31 |
|           | 8-bit   | 155.58   | 1069.20 |
|           | 4-bit (G=32)   | 196.80   | 862.69 |
| Llama-2-70B | Base    | OOM     ||
|           | 8-bit   | 19.13    | 1322.58 |
|           | 4-bit (G=32)   | 25.25    | 1097.66 |
| Llama-3.1-8B  | Base    |  93.89  | 1410.76 |
|           | 8-bit   | 137.64   | 1030.89 |
| Llama-3.1-70B | Base    | OOM     ||
|           | 8-bit   | 18.04    | 1253.78 |

### Speculative Sampling
[Verifier: Llama-70B (int4), Draft: Llama-7B (int4)](./scripts/speculate_70B_int4.sh): 48.4 tok/s

### Tensor Parallelism
| Model    | Number of GPUs | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | 1    |  104.9  | 1397.31 |
|           | 2   | 168.84   | 1181.99 |
|           | 4   | 254.02   | 955.83 |
|           | 8   | 328.43   | 704.10 |
| Llama-2-70B  | 1    |  OOM  |  |
|           | 2   | 21.32   | 1481.87 |
|           | 4   | 38.01   | 1340.76 |
|           | 8   | 62.50   | 1135.29 |
| Llama-3.1-8B  | 1    |  93.83  | 1408.37 |
|           | 2   | 149.10   | 1197.32 |
|           | 4   | 217.21   | 986.32  |
|           | 8   | 276.01   | 772.60 |
| Llama-3.1-70B  | 1    |  OOM  |  |
|           | 2   | 16.03   | 1130.81 |
|           | 4   | 37.45   | 1360.53 |
|           | 8   | 58.78   | 1129.61 |

### Tensor Parallelism + Quantization
| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-70B | Base    | 62.50     | 1135.29 |
|           | 8-bit   | 80.44    | 752.04 |
|           | 4-bit (G=32)   | 90.77    | 548.10 |
| Llama-3.1-70B | Base    | 58.78     | 1129.61 |
|           | 8-bit   | 75.58    | 726.57 |
| Llama-3.1-405B | 8-bit | 15.60 | 815.87 |

### AMD
Benchmarks run on one GCD of a MI-250x.

| Model    | Technique | Tokens/Second | Memory Bandwidth (GB/s) |
| -------- | ------- | ------ | ------ |
| Llama-2-7B  | Base    |  76.33  | 1028.70 |
|           | 8-bit   | 101.86   | 700.06 |

## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.

## Quantization
Choose device to use by
```bash
# The current support devices: cuda, cpu
export DEVICE=cuda
```
### Int8 Weight-Only Quantization
To generate this version of the model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int8.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```
To run with int8, just pass the int8 checkpoint to generate.py.
```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --device $DEVICE
```

### Int4 Weight-Only Quantization
To generate int4 version of model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4.g32.$DEVICE.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4 --groupsize 32
```

To run with int4, just pass the int4 checkpoint to generate.py.
```bash
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.pth --compile
```

## Speculative Sampling
To generate with speculative sampling (DRAFT_MODEL_REPO should point to a smaller model compared with MODEL_REPO).

In this example, the "smaller" model is just the int8 quantized version of the model.
```
export DRAFT_MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth
```

You can also run the draft model on a different device (e.g. `cuda:1`) and optionally add draft weight noise after load. The preferred configuration uses discrete noise **levels** with a shared level→std table (and supports per-layer configuration via `3*n_layer` level assignments).
```
export MODEL_DIR=checkpoints/modelscope/Llama-2-7b-chat-ms
python generate.py --compile --compile_prefill \
  --checkpoint_path $MODEL_DIR/model.pth \
  --draft_checkpoint_path $MODEL_DIR/model.pth \
  --device cuda:0 --draft_device cuda:1 \
  --draft_noise_level_stds 0 1e-3 \
  --draft_noise_levels 1 1 1 --draft_noise_seed 1234 \
  --speculate_k 5 --temperature 0
```

To use an INT8 weight-only target model but run the draft model in fp using dequantized INT8 weights (and optional noise), pass an INT8 checkpoint for both and enable `--draft_dequantize_int8`:
```
export MODEL_DIR=checkpoints/modelscope/Llama-2-7b-chat-ms
python generate.py --compile --compile_prefill \
  --checkpoint_path $MODEL_DIR/model_int8.pth \
  --draft_checkpoint_path $MODEL_DIR/model_int8.pth \
  --draft_dequantize_int8 \
  --device cuda:0 --draft_device cuda:1 \
  --draft_noise_level_stds 0 1e-3 \
  --draft_noise_levels 1 0 0 --draft_noise_seed 1234 \
  --speculate_k 5 --temperature 0
```
Legacy shortcut: `--draft_noise_std` is still supported (1 value or 3 values: FFN QKV OUT) and is applied uniformly across layers when level-based flags are not provided.
If you hit a `CUDA ... illegal memory access` inside `create_block_mask`, add `--no_compile_block_mask`.

To additionally quantize activations per-token and run int8xint8 matmuls for target linear layers, enable `--int8_act_quant`. To apply fake activation quantization to the draft model (still fp matmuls), enable `--draft_fake_act_quant_int8`.

Note: Running on an A100 80GB, albeit power-limited to 330 watts. Empirically, seems like peak bandwidth is about 1700 GB/s.

### Export calculator-compatible acceptance stats (`stats.json`)

To export a `selfspec-calculator` / `ppa-calculator` compatible acceptance histogram from a speculative run, pass `--stats_out`:

```bash
python generate.py --compile \
  --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth \
  --speculate_k 5 --temperature 0 \
  --stats_out out/my_run
```

This writes:
- `out/my_run/stats.json`
- `out/my_run/stats_meta.json` (disable with `--no_stats_meta`)

You can then run `ppa-calculator` (from `../selfspec-calculator`) directly on the exported file:

```bash
ppa-calculator \
  --model ../selfspec-calculator/examples/model.yaml \
  --hardware ../selfspec-calculator/examples/hardware.yaml \
  --stats out/my_run/stats.json \
  --prompt-lengths 64 128 256 \
  --output out/my_run/report.json
```

### Dataset aggregation (many prompts)

Use the dataset runner to aggregate acceptance across a prompt set (`.txt` = one prompt per line, `.jsonl` = `{"prompt": "..."}` by default):

```bash
python scripts/dataset_selfspec_stats.py \
  --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth \
  --prompts prompts.txt \
  --run_id my_dataset_run
```

Prompt-length sweep mode writes one file per length:

```bash
python scripts/dataset_selfspec_stats.py \
  --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model_int8.pth \
  --prompts prompts.jsonl \
  --prompt_lengths 64 128 256 \
  --run_id my_sweep
```


## Tensor Parallelism
```bash
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=2 generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```

## Experimental
### Evaluation
We use the EleutherAI evaluation harness to evaluate our model accuracy. To evaluate the accuracy, make sure the evaluation harness is installed and pass your model checkpoint and desired tasks to eval.py.

```bash
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile --tasks hellaswag winogrande
```

Note: Generative tasks are currently not supported for gpt-fast

Installation Instructions for the evaluation harness: https://github.com/EleutherAI/lm-evaluation-harness/tree/master#install

### GPTQ
We have a pure pytorch implementation of GPTQ that utilizes torch._dynamo.export to access the model structure. You can generate a GPTQ quantized
version of int4 quantization by using the same command to quantize it but adding 'gptq' to the quantization mode i.e.
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4-gptq.g32.pth
python quantize.py --mode int4-gptq --calibration_tasks wikitext --calibration_seq_length 2048
```

You can then eval or generate text with this model in the same way as above.

## Development Environment Specifications

### Python & Conda Environment
- **Conda Environment Name**: `gpt-fast`
- **Active Python Path**: `/home/zhaoyibo/anaconda3/envs/gpt-fast/bin/python`
- **Execution Rule**: Always use `conda run -n gpt-fast` for any python or pip commands.

### Network & Proxy Configuration
The server is located in a restricted network environment. **Before any task requiring internet access** (e.g., installing packages, hitting external APIs), you MUST enable the proxy.

- **Proxy Initialization**: 
    - Since `proxy_on` is a shell function, you must source the profile and execute the command in the same shell session:
    - Command: `source /etc/profile.d/clash.sh && proxy_on && [YOUR_COMMAND]`
- **Pre-task Checklist**: If a command fails due to connection timeout or network error, retry with the proxy initialization sequence prefix.
- **Verification**: You can verify connectivity by running: `source /etc/profile.d/clash.sh && proxy_on && curl -I https://www.google.com`

## License

`gpt-fast` is released under the [BSD 3](https://github.com/meta-pytorch/gpt-fast/main/LICENSE) license.

## Acknowledgements
Thanks to:
* Lightning AI for supporting pytorch and work in flash attention, int8 quantization, and LoRA fine-tuning.
* GGML for driving forward fast, on device inference of LLMs
* Karpathy for spearheading simple, interpretable and fast LLM implementations
* MLC-LLM for pushing 4-bit quantization performance on heterogeneous hardware
