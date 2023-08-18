# Data

The Vigogne models were trained on a variety of datasets, including open-source datasets, ChatGPT-distillation datasets (self-instruct, self-chat, and orca-style data), and translated datasets.

These datasets cover different purposes such as instruction-following and human-assistant chat.

<!-- ## Instruction-following Data

Here is a subset of the instruction-following data that was utilized to fine-tune the Vigogne-Instruct models:

|                    Dataset                    | Size  |                                                                Link                                                                 |                                                     Description                                                     |
| :-------------------------------------------: | :---: | :---------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|          Translated Stanford Alpaca           |  52k  | [alpaca_data_cleaned_fr_52k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/alpaca_data_cleaned_fr_52k.jsonl) |                   Stanford Alpaca data translated into French using `gpt-3.5-turbo` in April 2023                   |
|           French self-instruct data           | 100k  |                                                                 N/A                                                                 |   Instruction-following data generated using `gpt-3.5-turbo` in April 2023 (See [Self-Instruct](#self-instruct))    |
| French Databricks Dolly of Bactrian's version |  15k  |      [dolly_bactrian_fr_15k.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/instruct/dolly_bactrian_fr_15k.jsonl)      | French Dolly subset extracted from [`MBZUAI/Bactrian-X`](https://huggingface.co/datasets/MBZUAI/Bactrian-X) dataset |

## Chat Data

Here is a subset of the human-assistant chat data used to fine-tune the Vigogne-Chat models:

|            Dataset             | Size  |                                                          Link                                                           |                                                     Description                                                      |
| :----------------------------: | :---: | :---------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
|     French self-chat data      |  50k  |                                                           N/A                                                           |              Dialogue data generated using `gpt-3.5-turbo` in April 2023 (See [Self-Chat](#self-chat))               |
|  French dialogues from OASST1  |  1k   | [oasst_20230412_fr_top1.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/chat/oasst_20230412_fr_top1.jsonl) | French dialogues extracted from [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset |
| French dialogues from ShareGPT |  1k   |                  [sg_fr.jsonl](https://github.com/bofenghuang/vigogne/blob/main/data/chat/sg_fr.jsonl)                  |  French dialogues extracted from [RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K) dataset  | -->

## Data Collection

Below is a non-exhaustive list of datasets containing 🇫🇷 French instruction-following and chat examples (feel free to submit a PR to include more datasets 🤗):

- [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X)
- [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [ShareGPT](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)
- [camel-ai/ai_society_translated](https://huggingface.co/datasets/camel-ai/ai_society_translated)
- [Gt-Doremiti/gt-doremiti-instructions](https://huggingface.co/datasets/Gt-Doremiti/gt-doremiti-instructions)

## Data Generation

### Translate Alpaca Data

We used the [cleaned version](https://github.com/gururise/AlpacaDataCleaned) of the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and translated it into French using `gpt-3.5-turbo` through the chat completion API. The entire translation process cost about $40.

However, it's important to note that the translation may have affected the accuracy of certain tasks, such as generating rhyming words or correcting grammar (discussed [here](https://github.com/tloen/alpaca-lora/pull/127)). We welcome PRs to help improve the quality of this dataset!

To translate the Alpaca dataset, you can use the following command:

```bash
# Specify your OpenAI API key
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/translate_alpaca.py \
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
python scripts/data_generation/generate_instructions.py \
    --seed_tasks_path data/instruct/seed_tasks_vigogne.jsonl \
    --prompt_path data/instruct/prompt_vigogne.txt \
    --output_file data/instruct/self_instruct_data.jsonl \
    --num_instructions_to_generate 1 \
    --n_workers 1
```

### Self-Chat

We generated some multi-turn dialogues between a human user and an AI assistant by providing a topic to `gpt-3.5-turbo` and letting it generate a conversation, following the approach used in [Baize](https://arxiv.org/abs/2304.01196).

The following is an example of a provided topic and the generated dialogue:

```json
{
    "subject":"Quelle est la meilleure façon de lire un livre technique?",
    "messages":[
        {
            "role":"User",
            "content":"Je me demande quelle est la meilleure façon de lire un livre technique. As-tu des conseils à me donner ?"
        },
        {
            "role":"Assistant",
            "content":"Bien sûr, voici quelques conseils pour vous aider à lire un livre technique de manière efficace : \n\n1. Commencez par une lecture rapide pour vous familiariser avec le contenu du livre. \n2. Utilisez un surligneur ou prenez des notes pour marquer les parties importantes du texte. \n3. Ne passez pas trop de temps à essayer de comprendre chaque concept dès le début. Lisez plutôt le livre en entier et revenez aux parties plus difficiles plus tard. Vous aurez déjà une meilleure compréhension générale du sujet.\n4. Révisez régulièrement les parties que vous avez surlignées ou notées. relire les passages importants plusieurs fois pour vous assurer que vous les avez vraiment compris.\n5. N'hésitez pas à chercher des ressources supplémentaires pour vous aider à comprendre les parties les plus difficiles."
        },
        {
            "role":"User",
            "content":"C'est une bonne idée de prendre des notes et de surligner les parties importantes. Comment savoir quelles parties sont les plus importantes ?"
        },
        {
            "role":"Assistant",
            "content":"Les parties les plus importantes d'un livre technique sont généralement celles qui sont liées aux objectifs que vous vous êtes fixés en lisant ce livre. Si vous recherchez des informations sur un concept spécifique, vous voudrez peut-être surligner tous les passages qui y sont liés. Si vous voulez comprendre comment résoudre un problème particulier, vous voudrez peut-être surligner les parties qui offrent des solutions à ce problème. En général, c'est une bonne idée de surligner les parties qui expliquent les concepts clés et les méthodes de travail."
        }
    ]
}
```

Below is an example of a script we used to provide some translated subjects in [Quora](https://huggingface.co/datasets/quora) and have the model generate dialogues:

```bash
# Specify your OpenAI API key
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/generate_conversations.py \
    --input_json_file data/chat/subject_quora_fr_nllb3b3.jsonl \
    --output_json_file data/chat/self_chat_data_quora_fr.jsonl \
    --subject_field translated_subject \
    --output_subject_field subject \
    --id_prefix self-chat-quora- \
    --max_samples 1 \
    --max_parallel_requests 4
```

### Translate Dataset

We have provided a script to translate the dataset on the Hugging Face Hub. You need to specify the machine translation model to use and the data fields to translate.

```bash
python scripts/data_processing/translate_dataset.py \
    --dataset_name qwedsacf/grade-school-math-instructions \
    --output_file data/chat/grade_school_math_instructions_fr_nllb3b3.jsonl \
    --field_names '["INSTRUCTION", "RESPONSE"]' \
    --model_name_or_path facebook/nllb-200-3.3B
```

### Orca-Style Data

We generated instruction-following data in the style of [Orca](https://arxiv.org/abs/2306.02707), which introduces explanatory signals from the teacher model `gpt-4`, showcasing its reasoning process during response generation. Additionally, responses can also be generated using `gpt-3.5-turbo` as an intermediate teacher for curriculum learning.

Here's an example to illustrate:

```json
{
    "system": "Vous êtes un assistant IA qui aide les gens à trouver des informations. L'utilisateur vous posera une question. Votre tâche est de répondre aussi fidèlement que possible. Tout en répondant, réfléchissez étape par étape et justifiez votre réponse.",
    "messages": [
        {
            "content": "Utilisez le raisonnement pour apporter la réponse à la question suivante :\nLe petit salon de beauté de propriété locale a eu son grand ouverture, les gens espéraient qu'il va stimuler l'économie dans les environs de quoi?\nLes options :\nLe clergé\nLa boutique Barber\n- Le voisinage\n- la ville\nLe Strip Mall\nProcessus de raisonnement :",
            "role": "User"
        },
        {
            "content": "Pour répondre à cette question, analysons chaque option et déterminons celle qui correspond le mieux à la situation décrite.\n\n1. Le clergé: Le clergé fait référence aux membres du clergé religieux, il ne semble pas y avoir de lien direct entre un salon de beauté et une stimulation économique pour les membres du clergé. Donc, cette option n'est probablement pas correcte.\n\n2. La boutique Barber: Une boutique Barber fait souvent référence à un salon de coiffure masculin. Bien que ce soit lié à l'industrie de la beauté, l'ouverture d'un petit salon de beauté pourrait entraîner une compétition plutôt qu'une stimulation économique pour une boutique Barber. Cette option n'est donc pas la meilleure réponse non plus.\n\n3. Le voisinage: L'ouverture d'un petit salon de beauté pourrait attirer des clients dans le voisinage, stimulant ainsi l'économie locale, car les gens seraient plus susceptibles de dépenser de l'argent dans les environs. Cela semble être une réponse plausible.\n\n4. La ville: Même si l'ouverture d'un salon de beauté peut avoir un impact sur l'économie d'une ville, l'effet serait généralement moins direct et moins important que pour le voisinage immédiat. Cependant, cette réponse reste possible.\n\n5. Le Strip Mall: Un Strip Mall est un centre commercial composé de plusieurs magasins, généralement situé le long d'une artère principale. L'ouverture d'un petit salon de beauté pourrait stimuler l'économie d'un Strip Mall en attirant des clients pour d'autres magasins, mais cela dépend de la situation géographique du salon.\n\nEn prenant en compte ces informations, il semble que les deux options les plus plausibles soient \"Le voisinage\" et \"la ville\". Cependant, étant donné que la question mentionne spécifiquement \"les environs\", cela implique une portée géographique plus limitée. Par conséquent, la réponse la plus précise et justifiée serait \"Le voisinage\".",
            "role": "Assistant"
        }
    ]
}
```

To achieve this, we first translated the instructions in the [Flan-v2](https://github.com/google-research/FLAN/tree/main/flan/v2) collection into French using the script available in [Dataset Translation](#dataset-translation). You can access the sub-collections, namely [COT](https://huggingface.co/datasets/conceptofmind/cot_submix_original), [NIV2](https://huggingface.co/datasets/conceptofmind/niv2_submix_original), [T0](https://huggingface.co/datasets/conceptofmind/t0_submix_original), and [FLAN 2021](https://huggingface.co/datasets/conceptofmind/flan2021_submix_original), provided by @conceptofmind on the Hugging Face Hub.

Next, you can generate responses using the following script. Please note that the generation might take some time due to the rate limit, endpoint load, and the long text of query and response pairs.

```bash
# Specify your OpenAI API key
export OPENAI_API_KEY=YOUR/OPENAI/API/TOKEN

python scripts/data_generation/generate_responses.py \
    --input_json_file path/to/flanv2_translated.jsonl \
    --output_json_file path/to/flanv2_translated_completed.jsonl \
    --system_field system_prompt \
    --instruction_field translated_question \
    --response_field fr_response \
    --model gpt-4 \
    --max_parallel_requests 1 \
    --max_samples 1
```


### Translation Data

We've supplied a script for reformatting machine translation data into the instruction task format.

Here's an example:

```json
{
    "system": "You are an AI assistant that follows instructions extremely well. Help as much as you can.",
    "messages": [
        {
            "content": "Switch the specified sentences from their English form to French form.\n\nIt has been issued since 2006 to partner institutions and associations who have requested it and is on the Ministry's website, with each caption accompanied by an illustration.",
            "role": "User"
        },
        {
            "content": "Elle est diffusée depuis 2006 aux partenaires demandeurs (institutionnels et associatifs) et sur le site Internet du ministère, chaque phrase étant accompagnée d'une illustration.",
            "role": "Assistant"
        }
    ]
}
```

Here's an example of how we utilized the script to reformat the [WMT14](https://huggingface.co/datasets/wmt14) translation dataset. The instructions have been copied and adapted from the [Parrot](https://github.com/wxjiao/ParroT/blob/master/scripts/instruct_follow.txt) project.

```bash
python scripts/data_processing/prep_translation.py \
    --instruction_file_path scripts/data_processing/prompt_translation_en.txt \
    --output_file path/to/translation_task_wmt14_en_fr_cleaned.jsonl
```
