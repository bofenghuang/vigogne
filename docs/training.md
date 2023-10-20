# Training

## Supervised Fine-tuning

For efficient LLM fine-tuning, we use [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) from 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library. This involves freezing the base model's parameters and introducing a small number of learnable parameters.

For those with limited GPU memory, it's recommended to quantize certain computations to 8-bit or 4-bit precision using [LLM.int8()](https://arxiv.org/abs/2208.07339) or [QLoRA](https://arxiv.org/abs/2305.14314). Note that this might result in a slight training slowdown compared to the fp16 or bf16 versions.

Tools like [DeepSpeed](https://github.com/microsoft/DeepSpeed) or [FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api) are highly recommended for distributed learning. [FlashAttention](https://arxiv.org/abs/2307.08691) is essential for speeding up training and reducing memory usage with long sequences.

More examples can be found in [examples](https://github.com/bofenghuang/vigogne/blob/main/examples/train).

*Since version 2.2, I've refactored the training code, integrating specific elements inspired by the excellent training framework [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). Thanks to the Axolotl team for their contributions to the open-source community! The primary motivation behind maintaining my own framework is to have full control over the entire training process and customize it to my specific needs. I highly recommend using Axolotl for additional features.*
