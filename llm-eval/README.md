# OpenAI API Evaluation for `zhangyir/Copy_Benchmark`

This folder contains a simple OpenAI API evaluation script for the Hugging Face dataset:

```text
zhangyir/Copy_Benchmark
```

The script calls an OpenAI model through the Responses API and computes one accuracy value for each subset.

## Files

```text
eval_openai_copy_benchmark.py   # main evaluation script
README.md                       # this usage guide
```

## Scoring rule

The script follows the same evaluation logic as the benchmark eval code:

- `01-copy`: strict string match.
- `ab-copy`: extract only `a/A/b/B` from the model output, map `a/A -> a` and `b/B -> b`, then compare with the target.
- `python-list-conversion`: extract all numbers from the model output and the gold answer, then compare the full extracted number sequence.

API errors after all retry attempts are counted as incorrect.

## Install

```bash
pip install -U openai datasets huggingface_hub tqdm
```


## Note about Hugging Face loading on Windows

Some `datasets` versions on Windows may fail if a remote `hf://...` path is passed to `load_dataset`, because it can be interpreted as a local path. This script avoids that problem by first trying the standard `datasets.load_dataset(...)` call and then falling back to `huggingface_hub` file discovery plus `hf_hub_download(...)`.

## Set API key

Linux/macOS:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

You can also pass the key directly with `--api-key`, but the environment variable is safer.

## Quick sanity check

Run only the first 5 examples of `01-copy`:

```bash
python eval_openai_copy_benchmark.py \
  --model gpt-5.5 \
  --dataset zhangyir/Copy_Benchmark \
  --subset 01-copy \
  --limit 5
```

## Run one full subset

```bash
python eval_openai_copy_benchmark.py \
  --model gpt-5.5 \
  --dataset zhangyir/Copy_Benchmark \
  --subset ab-copy \
  --max-output-tokens 32768 \
  --temperature 0
```

Available subsets:

```text
01-copy
ab-copy
python-list-conversion
```

## Run all subsets

```bash
python eval_openai_copy_benchmark.py \
  --model gpt-5.5 \
  --dataset zhangyir/Copy_Benchmark \
  --subset all \
  --max-output-tokens 32768 \
  --temperature 0
```

## Resume an interrupted run

```bash
python eval_openai_copy_benchmark.py \
  --model gpt-5.5 \
  --dataset zhangyir/Copy_Benchmark \
  --subset ab-copy \
  --resume
```

Retry only examples whose latest row had an API error:

```bash
python eval_openai_copy_benchmark.py \
  --model gpt-5.5 \
  --dataset zhangyir/Copy_Benchmark \
  --subset ab-copy \
  --resume \
  --retry-errors
```

When `--resume --retry-errors` is used, older failed rows may remain in `predictions.jsonl`, but `summary.json` uses the latest row for each example id.

## Output files

For one subset, outputs are written to:

```text
openai_eval_outputs/<model>/<subset>/predictions.jsonl
openai_eval_outputs/<model>/<subset>/summary.json
```

For `--subset all`, an additional file is written to:

```text
openai_eval_outputs/<model>/summary_all.json
```

Example `summary.json`:

```json
{
  "subset": "01-copy",
  "model": "gpt-5.5",
  "num_examples": 5,
  "num_correct": 4,
  "num_api_errors": 0,
  "accuracy": 0.8
}
```

## Useful options

```text
--limit N                  evaluate only N examples per subset
--start N                  start from example index N
--max-attempts 3           retry budget for API errors
--timeout-sec 300          request timeout in seconds
--request-gap-sec 1        sleep between requests
--save-prompt              save prompt messages in predictions.jsonl
--temperature none         omit temperature from the API call
```

For long copy examples, keep `--max-output-tokens` large enough; otherwise the model may be truncated and receive an incorrect score.
