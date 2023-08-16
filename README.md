<p align="center" width="100%">
<img src="./assets/vigogne_logo.png" alt="Vigogne" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>

# Vigogne 🦙: French Instruction-following and Chat Models

<p align="center">
    <a href="https://github.com/bofenghuang/vigogne/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/bofenghuang/vigogne.svg">
    </a>
    <!-- <a href="https://github.com/bofenghuang/vigogne/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/bofenghuang/vigogne.svg?color=green">
    </a> -->
    <a href="https://github.com/bofenghuang/vigogne/blob/main/LICENSE">
        <img alt="Code License" src="https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg">
    </a>
    <a href="https://github.com/bofenghuang/vigogne/blob/main/DATA_LICENSE">
        <img alt="Data License" src="https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg">
    </a>
    <a href="https://huggingface.co/models?search=bofenghuang/vigogne">
        <img alt="Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow.svg">
    </a>
    <a href="https://colab.research.google.com/github/bofenghuang/vigogne/blob/main/notebooks/infer_chat.ipynb">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>

*The vigogne (French name for vicuña) is a South American camelid native to the Andes Mountains. It is closely related to the llama, alpaca, and guanaco.*

Vigogne is a collection of powerful 🇫🇷 French large language models (LLMs) that are open-source and designed for instruction-following and chat purposes.

The main contributions of this project include:

- Open-sourced 🦙 Vigogne models for French instruction-following and chat
- Efficient training code for fine-tuning LLMs such as [LLaMA](https://github.com/facebookresearch/llama), [Llama-2](https://ai.meta.com/llama), [Falcon](https://falconllm.tii.ae), and [FLAN-T5](https://huggingface.co/google/flan-t5-xl)
- Generated, translated, and collected French instruction-following and dialogue datasets, along with the used scripts
- Inference code, Gradio demo, and support for deploying within various ecosystems such as 🤗 Transformers, llama.cpp, FastChat, and vLLM

💡 *The screencast below shows the current 🦙 Vigogne-7B-Chat model running on Apple M1 Pro using 4GB of weights (no sped up).*

![](./assets/screencast_llamacpp_chat.gif)

## Table of Contents

- [Updates](#updates)
- [Installation](#installation)
- [🦙 Vigogne Models](#-vigogne-models)
- [Inference and Deployment](#inference-and-deployment)
- [Data](#data)
- [Training](#training)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Updates

- [2023/08/16]: Added support for serving using [FastChat](https://github.com/lm-sys/FastChat) and [vLLM](https://vllm.ai).
- [2023/08/02]: Implemented generation script for [Orca-style](https://arxiv.org/abs/2306.02707) data.
- [2023/07/31]: Integrated [FlashAttention](https://arxiv.org/abs/2307.08691) support and implemented training example packing.
- [2023/07/20]: Introduced the latest Vigogne models built upon [Llama-2](https://ai.meta.com/llama/use-policy).
- [2023/07/05]: Released Vigogne models based on [Falcon](https://falconllm.tii.ae)  and [MPT](https://www.mosaicml.com/blog/mpt-7b), with commercial-friendly licenses.
- [2023/06/05]: Integrated [QLoRA](https://arxiv.org/abs/2305.14314) support for improved training efficiency.
- [2023/05/15]: Introduced Vigogne-Chat models with enhanced conversational capabilities.
- [2023/05/11]: Implemented [Self-Chat](https://arxiv.org/abs/2304.01196) data generation script for conversational data.
- [2023/05/11]: Introduced improved Vigogne-Instruct-V2 models, trained on more diverse data.
- [2023/05/11]: Released annotated seed tasks in French and generation script for [Self-Instruct](https://arxiv.org/abs/2212.10560).
- [2023/04/03]: Expanded training scripts to incorporate seq2seq models.
- [2023/03/29]: Included deployment instructions using [llama.cpp](https://github.com/ggerganov/llama.cpp).
- [2023/03/26]: Released initial Vigogne-Instruct models trained on translated Stanford Alpaca data.
- [2023/03/26]: Open-sourced Vigogne project with optimized training scripts ([LoRA](https://arxiv.org/abs/2106.09685), [LLM.int8()](https://arxiv.org/abs/2208.07339)).

## Installation

1. Clone this repository

```bash
git clone https://github.com/bofenghuang/vigogne.git
cd vigogne
```

2. Install the package

```bash
# Install DeepSpeed if want to accelerate training with it
pip install deepspeed

# Install FlashAttention to further speed up training and reduce memory usage (essential for long sequences)
pip install packaging ninja
# For FlashAttention 1
# pip install --no-build-isolation flash-attn<2
# For FlashAttention 2
# Might takes 3-5 minutes on a 64-core machine
pip install --no-build-isolation flash-attn

pip install .
```

## 🦙 Vigogne Models

The fine-tuned 🦙 Vigogne models come in two types: **instruction-following models** and **chat models**. The instruction-following models are optimized to generate concise and helpful responses to user instructions, similar to `text-davinci-003`. Meanwhile, the chat models are designed for multi-turn dialogues, but they also perform well in instruction-following tasks, similar to `gpt-3.5-turbo`.

More information can be found in the [vigogne/model](docs/model.md).

## Inference and Deployment

This repository offers multiple options for inference and deployment, including Google Colab notebooks, Gradio demos, [FastChat](https://github.com/lm-sys/FastChat), and [vLLM](https://vllm.ai). It also offers guidance on conducting experiments using [llama.cpp](https://github.com/ggerganov/llama.cpp) on your personal computer.

More information can be found in the [vigogne/inference](docs/inference.md).

## Data

The Vigogne models were trained on a variety of datasets, including open-source datasets, ChatGPT-distillation datasets (self-instruct, self-chat, and orca-style data), and translated datasets.

More information can be found in the [vigogne/data](docs/data.md).

## Training

For efficient LLM fine-tuning, we utilize a technique called [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) from 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library. This approach involves freezing the base model's weights and introducing a small number of learnable parameters.

Additionally, for practitioners without access to GPUs with ample memory, it's advisable to consider quantizing certain computations to either 8-bit or 4-bit precision using [LLM.int8()](https://arxiv.org/abs/2208.07339) or [QLoRA](https://arxiv.org/abs/2305.14314). Be aware that this might lead to a minor reduction in speed compared to fp16 or bf16 versions.

We highly recommend the utilization of tools such as [DeepSpeed](https://github.com/microsoft/DeepSpeed) or [FSDP](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api), particularly when engaged in distributed learning scenarios. When dealing with long sequences, [FlashAttention](https://arxiv.org/abs/2307.08691) becomes crucial to speed up training and reduce memory usage.

More information can be found in the [vigogne/training](docs/training.md).

## Bias, Risks, and Limitations

Vigogne is still under development, and there are many limitations that have to be addressed. Please note that it is possible that the model generates harmful or biased content, incorrect information or generally unhelpful answers.

## Acknowledgements

Our project builds upon the following open-source projects for further development. We would like to extend our sincerest gratitude to the individuals involved in the research and development of these projects.

- [🤗 Transformers](https://github.com/huggingface/transformers) and [🤗 PEFT](https://github.com/huggingface/peft)
- [LLaMA](https://github.com/facebookresearch/llama)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Alpaca-LoRA by @tloen](https://github.com/tloen/alpaca-lora)
- [Baize](https://github.com/project-baize/baize-chatbot)
- [llama.cpp by @ggerganov](https://github.com/ggerganov/llama.cpp)

## Citation

If you find the model, data, and code in our project useful, please consider citing our work as follows:

```
@misc{vigogne,
  author = {Bofeng Huang},
  title = {Vigogne: French Instruction-following and Chat Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bofenghuang/vigogne}},
}
```
