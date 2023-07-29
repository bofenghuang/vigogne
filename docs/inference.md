# Inference and Deployment

This repository offers multiple options for inference and deployment, such as a **Google Colab notebook**, **Gradio demo**, and instructions for running experiments on your own PC using [**llama.cpp**](https://github.com/ggerganov/llama.cpp).

## Google Colab Notebook

You can use the following Google Colab Notebook to infer the Vigogne instruction-following models.

<a href="https://colab.research.google.com/github/bofenghuang/vigogne/blob/main/notebooks/infer_instruct.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

For the Vigogne-Chat models, refer to this notebook.

<a href="https://colab.research.google.com/github/bofenghuang/vigogne/blob/main/notebooks/infer_chat.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Gradio Demo

To launch a Gradio demo in streaming mode and interact with the Vigogne instruction-following models, simply execute the command provided below:

```bash
python vigogne/demo/demo_instruct.py --base_model_name_or_path bofenghuang/vigogne-7b-instruct
```

For the Vigogne-Chat models, you can follow this command:

```bash
python vigogne/demo/demo_chat.py --base_model_name_or_path bofenghuang/vigogne-7b-chat
```

## llama.cpp

The Vigogne models can now be easily deployed on PCs with the help of tools created by the community. The following instructions provide a detailed guide on how to combine Vigogne LoRA weights with the original LLaMA model, using [Vigogne-7B-Instruct](https://huggingface.co/bofenghuang/vigogne-7b-instruct) as an example. Additionally, you will learn how to quantize the resulting model to 4-bit and deploy it on your own PC using [llama.cpp](https://github.com/ggerganov/llama.cpp). For French-speaking users, you can refer to this excellent [tutorial](https://www.youtube.com/watch?v=BBf5h0HCFMY&t=292s&ab_channel=PereConteur) provided by @pereconteur.

**Note: the models will be quantized into 4-bit, so the performance might be worse than the non-quantized version. The responses are random due to the generation hyperparameters.**

Please ensure that the following requirements are met prior to running:

- As the models are currently fully loaded into memory, you will need adequate disk space to save them and sufficient RAM to load them. You will need at least 13GB of RAM to quantize the 7B model. For more information, refer to this [link](https://github.com/ggerganov/llama.cpp#memorydisk-requirements).
- It's best to use Python 3.9 or Python 3.10, as sentencepiece has not yet published a wheel for Python 3.11.

<!-- ### 1. Convert the original LLaMA model to the format used by Hugging Face

If you only have the weights of Facebook's original LLaMA model, you will need to convert it to the format used by Hugging Face. *Please skip this step if you have already converted the LLaMA model to Hugging Face's format or if you are using a third-party converted model from the Hugging Face model library, such as `decapoda-research/llama-7b-hf` and `huggyllama/llama-7b`. Please note that this project is not responsible for ensuring the compliance and correctness of using third-party weights that are not Facebook official.*

```bash
python scripts/convert_llama_weights_to_hf.py \
    --input_dir path/to/facebook/downloaded/llama/weights \
    --model_size 7B \
    --output_dir name/or/path/to/hf/llama/7b/model
```

### 2. Combine the LLaMA model with the Vigogne LoRA weights

```bash
# combine the LLaMA model in Hugging Face's format and the LoRA weights to get the full fine-tuned model
python scripts/export_state_dict_checkpoint.py \
    --base_model_name_or_path name/or/path/to/hf/llama/7b/model \
    --lora_model_name_or_path bofenghuang/vigogne-7b-instruct \
    --output_dir ./models/vigogne_7b_instruct \
    --base_model_size 7B

# download the tokenizer.model file
wget -P ./models https://huggingface.co/bofenghuang/vigogne-7b-instruct/resolve/main/tokenizer.model

# check the files
tree models
# models
# ├── vigogne_7b_instruct
# │   ├── consolidated.00.pth
# │   └── params.json
# └── tokenizer.model
``` -->

### 1. Convert the Vigogne model to the original LLaMA format

```bash
# convert the Vigogne model from Hugging Face's format to the original LLaMA format
python scripts/export_state_dict_checkpoint.py \
    --base_model_name_or_path bofenghuang/vigogne-7b-instruct \
    --output_dir ./models/vigogne_7b_instruct
    --base_model_size 7B

# download the tokenizer.model file
wget -P ./models https://huggingface.co/bofenghuang/vigogne-7b-instruct/resolve/main/tokenizer.model

# check the files
tree models
# models
# ├── vigogne_7b_instruct
# │   ├── consolidated.00.pth
# │   └── params.json
# └── tokenizer.model
```

### 2. Clone and build llama.cpp repo

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
# make with blas
# see https://github.com/ggerganov/llama.cpp#blas-build
```

### 3. Quantize the model

```bash
# convert the 7B model to ggml FP16 format
python convert.py path/to/vigogne/models/vigogne_7b_instruct

# quantize the model to 4-bits (using q4_0 method)
./quantize path/to/vigogne/models/vigogne_7b_instruct/ggml-model-f16.bin path/to/vigogne/models/vigogne_7b_instruct/ggml-model-q4_0.bin q4_0
```

### 4. Run the inference

```bash
# ./main -h for more information
./main -m path/to/vigogne/models/vigogne_7b_instruct/ggml-model-q4_0.bin --color -f path/to/vigogne/prompts/instruct.txt -ins -c 2048 -n 256 --temp 0.1 --repeat_penalty 1.1
```

For the Vigogne-Chat models, the previous steps for conversion and quantization remain the same. However, the final step requires a different command to run the inference.

```bash
./main -m path/to/vigogne/models/vigogne_7b_chat/ggml-model-q4_0.bin --color -f path/to/vigogne/prompts/chat.txt --reverse-prompt "<|UTILISATEUR|>:" --in-prefix " " --in-suffix "<|ASSISTANT|>:" --interactive-first -c 2048 -n -1 --temp 0.1
```

<!-- ## Text generation web UI

https://github.com/oobabooga/text-generation-webui

1. Clone and install the package

```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

2. Put the LLaMA model in Hugging Face's format inside the `models` folder

```bash
python download-model.py huggyllama/llama-7b
```

3. Put the Vigogne-7b-Instruct LoRA weights in the `lora` folder

```bash
git clone https://huggingface.co/bofenghuang/vigogne-7b-instruct .
```

4. Launch the web UI

```bash
# See https://github.com/oobabooga/text-generation-webui#starting-the-web-ui for more settings
python server.py --model huggyllama_llama-7b --lora vigogne-7b-instruct
```

## LlamaChat

https://github.com/alexrozanski/LlamaChat -->

