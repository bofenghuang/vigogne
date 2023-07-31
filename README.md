<p align="center" width="100%">
<img src="./assets/vigogne_logo.png" alt="Vigogne" style="width: 40%; min-width: 300px; display: block; margin: auto;">
</p>

# Vigogne 🦙: French Instruction-following and Chat Models

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/bofenghuang/vigogne/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/bofenghuang/vigogne/blob/main/DATA_LICENSE)
[![Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow.svg)](https://huggingface.co/models?search=bofenghuang/vigogne)

*The vigogne (French name for vicuña) is a South American camelid native to the Andes Mountains. It is closely related to the llama, alpaca, and guanaco.*

Vigogne is a collection of powerful French 🇫🇷 large language models (LLMs) that are open-source and designed for instruction-following and chat purposes.

The main contributions of this project include:

- Open-sourced 🦙 Vigogne models for French instruction-following and chat
- Efficient training code for fine-tuning LLMs such as [LLaMA](https://github.com/facebookresearch/llama), [Llama-2](https://ai.meta.com/llama), [Falcon](https://falconllm.tii.ae), and [FLAN-T5](https://huggingface.co/google/flan-t5-xl)
- Generated, translated, and collected French instruction-following and dialogue datasets, along with the used scripts
- Inference code and gradio demo, as well as detailed instructions for experiencing the quantized Vigogne models on your PC



💡 *The screencast below shows the current 🦙 Vigogne-7B-Chat model running on Apple M1 Pro using 4GB of weights (no sped up).*

![](./assets/screencast_llamacpp_chat.gif)

## Table of Contents

- [Updates](#updates)
- [Installation](#installation)
- [🦙 Vigogne Models](#-vigogne-models)
- [Inference and Deployment](#inference-and-deployment)
- [Data](#data)
- [Training](#training)
- [Example Outputs](#example-outputs)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Updates

- [2023/07/20]: Introduce the latest Vigogne models, based on Llama-2.
- [2023/07/05]: Released Vigogne models derived from Falcon and MPT, offering a more friendly license for commercial use.
- [2023/06/05]: Provided support for [QLoRA](https://arxiv.org/abs/2305.14314), resulting in a significant memory usage reduction.
- [2023/05/15]: Unveiled the Vigogne-Chat models and integrated the Gradio Demo for interactive chat demonstrations.
- [2023/05/11]: Added support for DeepSpeed, significantly improving the efficiency of model training.
- [2023/05/11]: Implemented chat data generation using [Self-Chat](https://arxiv.org/abs/2304.01196), empowering the model with more conversational abilities.
- [2023/05/11]: Presented the upgraded Vigogne-Instruct V2 models, trained on a larger and more diverse dataset.
- [2023/05/11]: Released annotated French seed tasks and the generation script for Self-Instruct.
- [2023/04/03]: Provided a training script for seq2seq models, utilizing LoRA and LLM.int8().
- [2023/03/29]: Included deployment instructions using [llama.cpp](https://github.com/ggerganov/llama.cpp).
- [2023/03/26]: Open-sourced the Vigogne projet and released fine-tuned LLaMA and BLOOM models trained on the translated Stanford Alpaca dataset.

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

pip install .
```

## 🦙 Vigogne Models

The fine-tuned 🦙 Vigogne models come in two types: **instruction-following models** and **chat models**. The instruction-following models are optimized to generate concise and helpful responses to user instructions, similar to `text-davinci-003`. Meanwhile, the chat models are designed for multi-turn dialogues, but they also perform well in instruction-following tasks, similar to `gpt-3.5-turbo`.

More information can be found in the [vigogne/model](docs/model.md).

## Inference and Deployment

This repository offers multiple options for inference and deployment, such as a **Google Colab notebook**, **Gradio demo**, and instructions for running experiments on your own PC using [**llama.cpp**](https://github.com/ggerganov/llama.cpp).

More information can be found in the [vigogne/inference](docs/inference.md).

## Data

The Vigogne models were trained using open-source datasets, ChatGPT-distillation datasets (self-instruct, self-chatting), and translated datasets.

More information can be found in the [vigogne/data](docs/data.md).

## Training

To fine-tune LLMs more efficiently, we employ a technique called [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) provided by 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library, which involves freezing the base model's weights and adding a small number of learnable parameters.

In addition, you can further reduce the memory usage during fine-tuning by using [LLM.int8()](https://arxiv.org/abs/2208.07339), which employs a 2-stage quantization method that quantizes part of the computation to int8. This enables efficient training on a single consumer GPU such as the RTX 4090. However, it may be slightly slower than the fp16 version. If your GPUs have enough memory, you can skip this step and train using [DeepSpeed](https://github.com/microsoft/DeepSpeed).

More information can be found in the [vigogne/training](docs/training.md).

## Example Outputs

Répondez à la question suivante : Les pratiques artistiques transforment-elles le monde ?

- **vigogne-7b-instruct**: Les pratiques d'art peuvent transformer le monde en créant des œuvres qui évoquent des sentiments, des idées et des émotions. Elles peuvent également être utilisées pour créer des Œuvres d'Art Vivant qui sont des œuvre vivantes qui peuvent être utilisés pour créé des expériences et des conversations. Les praticiens d'arts peuvent également utiliser leurs pratiques pour créerer des œufs d'âme, des êtres vivants créés à partir de matériaux naturels, qui peuvent avoir un impact sur la façon dont les gens pensent et ressentent le monde.

<details><summary>Other models</summary>

- **Alpaca-LoRA**: Yes, artistic practices can transform the world. Art can be used to express ideas, explore new perspectives, and create dialogue. Art can also be used to inspire and motivate people to take action, and to create positive change in their communities.

- **text-davinci-003**: Oui, les pratiques artistiques peuvent transformer le monde. Les artistes peuvent utiliser leur art pour exprimer leurs opinions et leurs idées, et ces idées peuvent avoir un impact sur la société et le monde. Les artistes peuvent également utiliser leur art pour sensibiliser le public à des sujets importants et pour encourager le changement. Les pratiques artistiques peuvent donc être un puissant outil pour transformer le monde.

</details>

---

Expliquez la différence entre DoS et phishing en français.

- **vigogne-7b-instruct**: Le DoS (Denial of Service) est une attaque informatique qui consiste à bloquer ou à interrompre l'accès à un service ou à un système informatique. Le phishing est une technique de fraude qui consisterait à obtenir des informations sensibles d'un utilisateur en lui demandant de fournir des renseignements, tels que des mots de passe ou des numéros de carte de crédit.

<details><summary>Other models</summary>

- **Alpaca-LoRA**: In French, the difference between a DDoS attack and phishing is that a DDoS attack is an attempt to overwhelm a system with a large amount of traffic, while phishing is an attempt to steal personal information by posing as a trusted source.

- **text-davinci-003**:

DoS (Denial of Service) est une attaque qui vise à rendre indisponible des services web ou réseau en saturant un serveur ou un réseau avec des requêtes. Elle peut aussi être utilisée pour perturber un réseau en envoyant des paquets mal formés ou en le surchargant.

Phishing est une forme d'escroquerie en ligne visant à voler des informations personnelles telles que des mots de passe et des numéros de carte de crédit en incitant les victimes à fournir leurs informations sur des faux sites Web ou des emails frauduleux.

</details>

---

Résumez le texte suivant en une seule phrase.

La science des données est l'étude des données afin d'en extraire des informations significatives pour les entreprises. Il s'agit d'une approche pluridisciplinaire qui combine des principes et des pratiques issus des domaines des mathématiques, des statistiques, de l'intelligence artificielle et du génie informatique, en vue d'analyser de grands volumes de données. Cette analyse aide les scientifiques des données à poser des questions et à y répondre, comme Que s'est-il passé, Pourquoi cela s'est-il passé, Que va-t-il se passer et Que peut-on faire avec des résultats.

- **vigogne-7b-instruct**: Le scientifique des données utilise des données pour analyser et extraire de l’information pour aider les entrepreneurs à posent des questions, répondre à des questions existantes et prendre des décisions.

<details><summary>Other models</summary>

- **Alpaca-LoRA**: La science des données est l'étude des données afin d'en extraire des informations significatives pour les entreprises.

- **text-davinci-003**: La science des données combine des principes et des pratiques issus des domaines des mathématiques, des statistiques, de l'intelligence artificielle et du génie informatique pour analyser de grands volumes de données afin d'en extraire des informations significatives pour les entreprises.

</details>

---

Écrivez une fonction qui prend une liste de chaînes de caractères et renvoie une liste sans doublons.

- **vigogne-7b-instruct**:

```python
def remove_duplicates(lst):
    result = []
    for item in lst:
        if item not in result:
            result.append(item)
    return result
```

<details><summary>Other models</summary>

- **Alpaca-LoRA**:

```python
def remove_duplicates(list):
    seen = set()
    return [x for x in list if x not in seen]
```

- **text-davinci-003**:

```python
def remove_duplicates(liste):
  new_list = []
  for item in liste:
    if item not in new_list:
      new_list.append(item)
  return new_list

liste = ["a", "b", "c", "a", "d", "e", "f", "c"]
print(remove_duplicates(liste)) # ['a', 'b', 'c', 'd', 'e', 'f']
```

</details>

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
