# 2D-RoPE

Links:

- Data: [https://huggingface.co/datasets/zhangyir/Copy](https://huggingface.co/datasets/zhangyir/Copy)
- Paper: (upcoming)
- Checkpoints: (upcoming)

This is the official source code for the paper **"Frontier Language Models Struggle to Copy: Text Can Be Better Viewed In 2D"** ([https://arxiv.org/abs/xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)).

## Code Structure

The code is separated into multiple folders:

- `pretraining/`: Implementation for the pretraining experiments of LLMs.
- `finetuning/`: Code for finetuning HuggingFace formatted checkpoints on the Binary Copy tasks.
- `data-generation/`: Code for generating the Binary Copy task data.
- `synthetic-experiments`: Code for the synthetic experiments of various small-scale language models on the Binary Copy tasks.

## How to Cite?

You can cite us with the following BibTeX.

```bibtex
@inproceedings{wen2026frontier,
    title={Frontier Language Models Struggle to Copy: Text Can Be Better Viewed in 2D},
    author={Haodong Wen and Yiran Zhang and Yingfa Chen and Kaifeng Lyu},
    booktitle={ICML 2026 Workshop on Structured Probabilistic Inference {\&} Generative Modeling},
    year={2026},
    url={https://openreview.net/forum?id=r3UVsj13Mr}
}
```
