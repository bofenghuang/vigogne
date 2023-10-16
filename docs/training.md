# Training

## Supervised Fine-tuning

For efficient LLM fine-tuning, we utilize a technique called [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) from 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library. This approach involves freezing the base model's weights and introducing a small number of learnable parameters.

Additionally, for practitioners without access to GPUs with ample memory, it's advisable to consider quantizing certain computations to either 8-bit or 4-bit precision using [LLM.int8()](https://arxiv.org/abs/2208.07339) or [QLoRA](https://arxiv.org/abs/2305.14314). Be aware that this might lead to a minor reduction in speed compared to fp16 or bf16 versions.

We highly recommend the utilization of tools such as [DeepSpeed](https://github.com/microsoft/DeepSpeed) or [FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api), particularly when engaged in distributed learning scenarios. When dealing with long sequences, [FlashAttention](https://arxiv.org/abs/2307.08691) becomes crucial to speed up training and reduce memory usage.

More examples can be found in [examples](https://github.com/bofenghuang/vigogne/blob/main/examples/train).

Since version 3.0, I've refactored the training code, incorporating specific elements inspired by the fantastic training framework [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl). Thanks to the Axolotl team for their valuable contributions to the open-source community! The primary motivation behind maintaining my own framework is to have full control over the entire training process and customize it to my specific needs. I highly recommend using their framework for additional features.
