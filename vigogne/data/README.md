# Data

The Vigogne models were trained on a variety of datasets, including open-source datasets, ChatGPT-distillation datasets (self-instruct and self-chatting), and translated datasets.

These datasets cover different purposes such as instruction-following and human-assistant chat.

## Instruction-following Data

Here is a subset of the instruction-following data that was utilized to fine-tune the Vigogne-Instruct models:

|                    Dataset                    |                   Size                    |                                                                       Link                                                                        |                                                                                            Description                                                                                            |
| :-------------------------------------------: | :---------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|          Translated Stanford Alpaca           |                    52k                    |        [alpaca_data_cleaned_fr_52k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/alpaca_data_cleaned_fr_52k.jsonl)        |                                                          Stanford Alpaca data translated into French using `gpt-3.5-turbo` in April 2023                                                          |
|           French self-instruct data           |                   100k                    |                                                                        N/A                                                                        |                                          Instruction-following data generated using `gpt-3.5-turbo` in April 2023 (See [Self-instruct](#self-instruct))                                           |
| French Databricks Dolly of Bactrian's version |                    15k                    |             [dolly_bactrian_fr_15k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/dolly_bactrian_fr_15k.jsonl)             |                                        French Dolly subset extracted from [`MBZUAI/Bactrian-X`](https://huggingface.co/datasets/MBZUAI/Bactrian-X) dataset                                        |
|                     <!--                      | Translated Grade School Math Instructions |                                                                        8k                                                                         |              [grade_school_math_instructions_fr_nllb3b3_8k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/grade_school_math_instructions_fr_nllb3b3_8k.jsonl)              | [`qwedsacf/grade-school-math-instructions`](https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions) dataset translated into French using [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) |
|          Translated Logic Inference           |                    54k                    | [logic_inference_oa_fr_nllb3b3_54k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/logic_inference_oa_fr_nllb3b3_54k.jsonl) | [`KK04/LogicInference_OA`](https://huggingface.co/datasets/KK04/LogicInference_OA) dataset translated into French using [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) | -->                                                                                                                                                                                                                                 |

## Chat Data

Here is a subset of the human-assistant chat data used to fine-tune the Vigogne-Chat models:

|                          Dataset                          |                  Size                  |                                                                   Link                                                                    |                                                                     Description                                                                     |
| :-------------------------------------------------------: | :------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------: |
|                 French self-chatting data                 |                  50k                   |                                                                    N/A                                                                    |                          Dialogue data generated using `gpt-3.5-turbo` in April 2023 (See [Self-chatting](#self-chatting))                          |
|               French dialogues from OASST1                |                   1k                   |          [oasst_20230412_fr_top1.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/chat/oasst_20230412_fr_top1.jsonl)          |                French dialogues extracted from [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset                 |
|              French dialogues from ShareGPT               |                   1k                   |                           [sg_fr.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/chat/sg_fr.jsonl)                           |                 French dialogues extracted from [RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) dataset                  |
|                           <!--                            | Reformatted translated Stanford Alpaca |                                                                    52k                                                                    | [converted_alpaca_data_cleaned_fr_52k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/chat/converted_alpaca_data_cleaned_fr_52k.jsonl) | Translated Stanford Alpaca dataset reformatted into dialogues |
|           Reformatted French self-instruct data           |                  100k                  |                                                                    N/A                                                                    |                                              Translated self-instruct data reformatted into dialogues                                               |
| Reformatted French Databricks Dolly of Bactrian's version |                  15k                   | [converted_dolly_bactrian_fr_15k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/chat/converted_dolly_bactrian_fr_15k.jsonl) |                                Translated Databricks Dolly dataset of Bactrian's version reformatted into dialogues                                 | -->                                                           |

## Data Preparation

### Translate Alpaca Data

We used the [cleaned version](https://github.com/gururise/AlpacaDataCleaned) of the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and translated it into French using `gpt-3.5-turbo` through the chat completion API. The entire translation process cost about $40.

However, it's important to note that the translation may have affected the accuracy of certain tasks, such as generating rhyming words or correcting grammar (discussed [here](https://github.com/tloen/alpaca-lora/pull/127)). We welcome PRs to help improve the quality of this dataset!

To translate the Alpaca dataset, you can use the following command:

```bash
# Specify your OpenAI API key
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python vigogne/data/translate_alpaca.py \
    --input_file data/alpaca_data_cleaned.jsonl \
    --output_file data/alpaca_data_cleaned_fr.jsonl \
    --failed_output_file data/alpaca_data_cleaned_fr_failed.jsonl \
    --model gpt-3.5-turbo \
    --max_parallel_requests 16
```

### Self-Instruct

Since the quality of instruction-following tasks translated from English to French is not ideal, we also generated some French instruction-following tasks directly using the data generation pipeline from the [self-instruct paper](https://arxiv.org/abs/2212.10560) and [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca). 

- We manually translated the [175 English seed tasks](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl) from Stanford Alpaca into French, and also made efforts to remove cultural biases in the translations. For example, in the instruction "Suggest some games that can be played by a group of people.", we replaced the list of games (e.g., Balderdash, Pictionary) with games more commonly played in France (e.g., Menteur, Trivial Pursuit). You can find the translated seed tasks in [`seed_tasks_vigogne.jsonl`](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/seed_tasks_vigogne.jsonl).
- We translated and adapted the prompt ([`prompt_vigogne.txt`](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/prompt_vigogne.txt)) that specifies the requirements for instruction-following data generation to `gpt-3.5-turbo`.
- We adopted batch generation to reduce the API cost by randomly sampling 3 seed tasks and generating 17 additional instruction-following tasks.
- Launch with multiple threads to speed up the process, but keep in mind the [upper limit](https://platform.openai.com/docs/guides/rate-limits/overview) of the OpenAI API.

You can use the following script to generate the instruction-following data:

```bash
# Specify your OpenAI API key
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

# num_instructions_to_generate is by worker
python scripts/generate_instruction.py \
    --seed_tasks_path data/instruct/seed_tasks_vigogne.jsonl \
    --prompt_path data/instruct/prompt_vigogne.txt \
    --output_file data/instruct/self_instruct_data.jsonl \
    --num_instructions_to_generate 1 \
    --n_workers 1
```

### Self-Chatting

We generated some multi-turn dialogues between a human user and an AI assistant by providing a topic to `gpt-3.5-turbo` and letting it generate a conversation, following the approach used in [Baize](https://arxiv.org/abs/2304.01196).

The following is an example of a provided topic and the generated dialogue:

```json
{
    "subject":"Quelle est la meilleure façon de lire un livre technique?",
    "conversation":[
        {
            "role":"UTILISATEUR",
            "content":"Je me demande quelle est la meilleure façon de lire un livre technique. As-tu des conseils à me donner ?"
        },
        {
            "role":"ASSISTANT",
            "content":"Bien sûr, voici quelques conseils pour vous aider à lire un livre technique de manière efficace : \n\n1. Commencez par une lecture rapide pour vous familiariser avec le contenu du livre. \n2. Utilisez un surligneur ou prenez des notes pour marquer les parties importantes du texte. \n3. Ne passez pas trop de temps à essayer de comprendre chaque concept dès le début. Lisez plutôt le livre en entier et revenez aux parties plus difficiles plus tard. Vous aurez déjà une meilleure compréhension générale du sujet.\n4. Révisez régulièrement les parties que vous avez surlignées ou notées. relire les passages importants plusieurs fois pour vous assurer que vous les avez vraiment compris.\n5. N'hésitez pas à chercher des ressources supplémentaires pour vous aider à comprendre les parties les plus difficiles."
        },
        {
            "role":"UTILISATEUR",
            "content":"C'est une bonne idée de prendre des notes et de surligner les parties importantes. Comment savoir quelles parties sont les plus importantes ?"
        },
        {
            "role":"ASSISTANT",
            "content":"Les parties les plus importantes d'un livre technique sont généralement celles qui sont liées aux objectifs que vous vous êtes fixés en lisant ce livre. Si vous recherchez des informations sur un concept spécifique, vous voudrez peut-être surligner tous les passages qui y sont liés. Si vous voulez comprendre comment résoudre un problème particulier, vous voudrez peut-être surligner les parties qui offrent des solutions à ce problème. En général, c'est une bonne idée de surligner les parties qui expliquent les concepts clés et les méthodes de travail."
        }
    ]
}
```

Below is an example of a script we used to provide some translated subjects in [Quora](https://huggingface.co/datasets/quora) and have the model generate dialogues:

```bash
# Specify your OpenAI API key
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python vigogne/data/generate_conversations.py \
    --input_json_file data/chat/subject_quora_fr_nllb3b3.jsonl \
    --output_json_file data/chat/self_chat_data_quora_fr.jsonl \
    --subject_field translated_subject \
    --output_subject_field subject \
    --id_prefix self-chat-quora- \
    --max_samples 1 \
    --max_parallel_requests 4
```

### Translate dataset

We have provided a script to translate the dataset on the Hugging Face Hub. You need to specify the machine translation model to use and the data fields to translate.

```bash
python vigogne/data/translate_dataset.py \
    --dataset_name qwedsacf/grade-school-math-instructions \
    --output_file data/chat/grade_school_math_instructions_fr_nllb3b3.jsonl \
    --field_names '["INSTRUCTION", "RESPONSE"]' \
    --model_name_or_path facebook/nllb-200-3.3B
```
