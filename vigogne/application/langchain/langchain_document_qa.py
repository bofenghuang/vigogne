#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Bofeng Huang

import logging
from functools import partial
from typing import Optional

import fire
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.llms import VLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger("__name__")

# system_message = "Vous êtes un assistant IA qui répond à la question posée à la fin en utilisant le contexte suivant. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse."
stuff_prompt_template = """<|system|>: Vous êtes un assistant IA qui répond à la question posée à la fin en utilisant le contexte suivant. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
<|user|>: {context}
Question: {question}
<|assistant|>:"""
refine_initial_prompt_template = """<|system|>: Vous êtes un assistant IA qui répond à la question posée à la fin en utilisant le contexte suivant. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
<|user|>: {context_str}
Question: {question}
<|assistant|>:"""
refine_prompt_template = """<|system|>: Vous êtes un assistant IA qui répond à la question posée à la fin en utilisant le contexte suivant. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
<|user|>: Question: {question}
Réponse existante: {existing_answer}
Vous avez la possibilité d'affiner la réponse existante (seulement si nécessaire) avec plus de contexte ci-dessous.
------------
{context_str}
------------
Veuillez affiner la réponse existante avec le contexte fournie. Si le contexte n'est pas utile, renvoyez la réponse existante.
<|assistant|>:"""
# stuff_prompt_template = partial(prompt_template.format, system_message=system_message)


def main(
    input_file: Optional[str] = None,
    web_url: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    chain_type: str = "stuff",
    embedding_model_name_or_path: str = "dangvantuan/sentence-camembert-base",
    llm_model_name_or_path: str = "bofenghuang/vigogne-2-7b-chat",
    llm_revision: str = "main",
    initial_question: Optional[str] = None,
):
    """
    Simple example for QA over Documents. Check out for more details on https://python.langchain.com/docs/use_cases/question_answering

    Args:
        input_file (str):
            Path to the input file.
        chain_type (str):
            Chain type for passing documents to LLM.
            More details on https://python.langchain.com/docs/modules/chains/document
        embedding_model_name_or_path (str):
            Name or path to sentence-transformers embedding model.
            - Multilingual: paraphrase-multilingual-mpnet-base-v2, paraphrase-multilingual-MiniLM-L12-v2
            - French: dangvantuan/sentence-camembert-base, dangvantuan/sentence-camembert-large
        llm_model_name_or_path (str):
            Name or path to Hugging Face LLM.
    """

    if chain_type not in {"stuff", "refine"}:
        raise ValueError(f"Invalid value for chain_type: {chain_type}")

    # Step 1: load
    logger.info("Loading text...")
    if input_file:
        loader = TextLoader(input_file)
    elif web_url:
        loader = WebBaseLoader(web_url)
    else:
        raise ValueError("You need specify either `input_file` or `web_url` to load documents")
    data = loader.load()

    # Step 2: split
    logger.info("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(data)

    # load embedding model
    logger.info("Loading embedding model...")
    # OpenAI embedding
    # embedding_model = OpenAIEmbeddings()
    # HF embedding
    # model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name_or_path,
        # model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Step 3: store
    logger.info("Storing chunks...")
    vectorstore = FAISS.from_documents(all_splits, embedding_model)

    # load LLM
    logger.info("Loading LLM...")
    # load from openai
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # load HF model
    llm = HuggingFacePipeline.from_model_id(
        model_id=llm_model_name_or_path,
        task="text-generation",
        device=0,
        model_kwargs={
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        },
        pipeline_kwargs={
            "revision": llm_revision,
            # "trust_remote_code": True,
            "temperature": 0.1,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.1,
        },
    )
    # todo: load llm using vllm
    # llm = VLLM(
    #     model=llm_model_name_or_path,
    #     trust_remote_code=True,
    #     dtype="float16",
    #     temperature=0.1,
    #     max_new_tokens=1024,
    #     # top_k=10,
    #     # top_p=0.95,
    # )
    # todo: load llm using llama.cpp

    # Step 4&5: retrieve and generate
    if chain_type == "stuff":
        prompt = PromptTemplate(template=stuff_prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": prompt}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
        )
    elif chain_type == "refine":
        question_prompt = PromptTemplate(
            template=refine_initial_prompt_template,
            input_variables=["context_str", "question"],
        )
        refine_prompt = PromptTemplate(
            template=refine_prompt_template,
            input_variables=["question", "existing_answer", "context_str"],
        )
        chain_type_kwargs = {"question_prompt": question_prompt, "refine_prompt": refine_prompt}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="refine",
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
            # return_intermediate_steps=True,
        )

    if initial_question:
        result = qa_chain({"query": initial_question})
        logger.info(f"Initial query: \n{result['query'].strip()}")
        logger.info(f"Response: \n{result['result'].strip()}")

    while True:
        question = input("Veuillez poser une question: ")
        result = qa_chain({"query": question})
        logger.info(f"Query: \n{result['query'].strip()}")
        logger.info(f"Response: \n{result['result'].strip()}")
        # logger.info(f"Source documents: \n{result['source_documents'].strip()}")


if __name__ == "__main__":
    fire.Fire(main)
