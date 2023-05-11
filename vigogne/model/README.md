# Models

The fine-tuned 🦙 Vigogne models come in two types: **instruction-following models** and **chat models**. The instruction-following models are optimized to generate concise and helpful responses to user instructions, similar to `text-davinci-003`. Meanwhile, the chat models are designed for multi-turn dialogues, but they also perform well in instruction-following tasks, similar to `gpt-3.5-turbo`.

<!-- todo -->
**User Notice**: Facebook has not made the official LLaMA model weights open source, although various third-party download links are available online, such as `decapoda-research/llama-7b-hf` and `huggyllama/llama-7b` in the Hugging Face model library. It should be noted that the use of these links may not comply with Facebook's policies. Due to the reasons mentioned above, the project cannot release the complete weights of fine-tuned models. However, only the LoRA weights can be provided, which can be considered as a "patch" for the original LLaMA model.

### Instruction-following Models

The Vigogne-Instruct models were fine-tuned using approximately 224k instruction-following tasks. Additional details regarding the data are available in the [vigogne/data](../data/README.md).

The LoRA weights for the Vigogne-Instruct models can be found on the 🤗 Hugging Face Hub:

|                                                  Model                                                  |                              Description                               |
| :-----------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------: |
|        [bofenghuang/vigogne-instruct-7b](https://huggingface.co/bofenghuang/vigogne-instruct-7b)        |          LLaMA-7B model fine-tuned for instruction-following           |
|       [bofenghuang/vigogne-instruct-13b](https://huggingface.co/bofenghuang/vigogne-instruct-13b)       | LLaMA-13B model fine-tuned for instruction-following (**Recommended**) |
| [bofenghuang/vigogne-instruct-bloom-7b1](https://huggingface.co/bofenghuang/vigogne-instruct-bloom-7b1) |          BLOOM-7B1 model fine-tuned for instruction-following          |

The graph below illustrates the loss curve for the fine-tuning process, based on 5000 evaluation examples.

<img src="../../assets/sft_instruct_eval_loss.png" style="width: 60%;">

<!-- ### Chat Models

The Vigogne-Chat models were fine-tuned using approximately 273k dialogue examples. Additional details regarding the data are available in the [vigogne/data](../data/README.md).

The LoRA weights for the Vigogne-Instruct models can be found on the 🤗 Hugging Face Hub:

| Model |            Description             |
| :---: | :--------------------------------: |
| TODO  | LLaMA-7B model fine-tuned for chat |
 -->
