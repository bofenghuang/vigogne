# Application

This repository provides integration examples for incorporating Vigogne models into diverse application ecosystems, including [LangChain](https://github.com/langchain-ai/langchain).

## 🦜️🔗 LangChain

[LangChain](https://github.com/langchain-ai/langchain) is a framework designed to create applications powered by LLMs for various end-to-end use cases such as chatbots, sourcing-based question answering, and structured data analysis.

Install LangChain with:

```bash
pip install langchain
```

### Q&A over Documents

LangChain leverages the advanced text comprehension capabilities of LLMs to execute automated Q&A tasks on targeted documents. This is achieved through a comprehensive pipeline including text loading, segmentation, vectorization, storage, retrieval, and response generation when presented with a question and relevant retrieved context.

Further details can be found in the LangChain [documentation](https://python.langchain.com/docs/use_cases/question_answering).

Here's an example that utilizes [dangvantuan/sentence-camembert-base](https://huggingface.co/dangvantuan/sentence-camembert-base) as embedding model, [Faiss](https://github.com/facebookresearch/faiss) as efficient similarity search engine, and leverages Vigogne to generate responses based on the query and the retrieved context.

```bash
# Install requirements
# pip install -U sentence_transformers faiss-gpu

# Run QA on local file
python vigogne/application/langchain/langchain_document_qa.py \
    --input_file "/path/to/your/input/file" \
    --embedding_model_name_or_path "dangvantuan/sentence-camembert-base" \
    --llm_model_name_or_path "bofenghuang/vigogne-2-7b-chat" \

# Run QA on web page
python vigogne/application/langchain/langchain_document_qa.py \
    --web_url "https://zaion.ai/en/resources/zaion-lab-blog/zaion-emotion-dataset" \
    --embedding_model_name_or_path "dangvantuan/sentence-camembert-base" \
    --llm_model_name_or_path "bofenghuang/vigogne-2-7b-chat" \
    --initial_question "Donne la définition de la speech emotion diarization."
# Output:
# La Speech Emotion Diarization (SED) est une tâche proposée pour la reconnaissance fine des émotions dans les discours. 
# Elle vise à déterminer si des émotions spécifiques sont présentes dans une phrase et à identifier leurs limites respectives. 
# Comparativement aux méthodes traditionnelles de reconnaissance d'émotions dans les phrases (SER), SED offre une approche plus précise et détaillée en identifiant les limites temporelles des émotions. 
# Le Zaion Emotion Dataset (ZED) est un ensemble de données annotées avec des étiquettes d'émotions discretes et des limites d'émotions au niveau du cadre pour chaque phrase parlée.
```

