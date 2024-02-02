import json
import os
import csv
from typing import (
    Any,
    List,
    Tuple
)

import marqo
import openai
from langchain.docstore.document import Document
from langchain_community.vectorstores.marqo import Marqo
from ragas.langchain import RagasEvaluatorChain
from ragas.llms import LangchainLLM
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy, AnswerCorrectness, AnswerSimilarity, AnswerRelevancy
)
from ragas.metrics.critique import harmfulness
from ragas.embeddings import HuggingfaceEmbeddings

from configparser import ConfigParser

config_file_path = 'config.ini'  # Update with your config file path
config = ConfigParser()
config.read(config_file_path)

# metrics
metrics = [
    faithfulness,
    context_recall,
    harmfulness,
    context_relevancy,
    context_precision,
]

# csv header
fieldnames = ['query', 'harmfulness_score', 'faithfulness_score', 'context_relevancy_score', "context_recall_score", "context_precision_score"]

marqo_url = os.getenv("MARQO_URL", None)
marqoClient = marqo.Client(url=marqo_url)

vectorstore = Marqo(marqoClient, "sakhi_teacher_activities_flaxbase", searchable_attributes=["text"])


def get_score_filtered_documents(search_documents: List[Tuple[Document, Any]], min_score=0.0):
    return [(document, search_score) for document, search_score in search_documents if search_score > min_score]


def get_filtered_documents(search_documents: List[Tuple[Document, Any]]):
    return [document for document, search_score in search_documents]


def get_formatted_documents(filter_documents: List[Tuple[Document, Any]]):
    sources = ""
    for document, _ in filter_documents:
        sources += f"""
            > {document.page_content} \n Source: {document.metadata['file_name']},  page# {document.metadata['page_label']};\n\n
            """
    return sources


evaluations = json.loads(config.get('benchmark', 'evaluations', fallback=None))
with open('benchmark_results.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    rows = []
    for user_query, ground_truths in evaluations.items():
        row = {"query": user_query}
        documents = vectorstore.similarity_search_with_score(user_query, k=20)
        filtered_documents = get_score_filtered_documents(documents, 0.7)
        filtered_documents = filtered_documents[:2]
        contexts = get_formatted_documents(filtered_documents)
        source_documents = get_filtered_documents(filtered_documents)

        system_rules = """You are a simple AI assistant specially programmed to help a teacher with learning and teaching materials for development of children in the age group of 3 to 8 years. Your knowledge base includes only the given documents.
            Guidelines:
                - Always pick relevant 'documents' for the given 'question'. Ensure that your response is directly based on the relevant documents from the given documents.
                - Your answer must be firmly rooted in the information present in the relevant documents.
                - Your answer should be in very simple English, for those who may not know English well.
                - Your answer should not exceed 200 words.
                - Always return the 'Source' of the relevant documents chosen in the 'answer' at the end.
                - answer format should strictly follow the format given in the 'Example of answer' section below.
                - If no relevant document is given, then you should answer "I'm sorry, but I am not currently trained with relevant documents to provide a specific answer for your question.'.
                - If the question is “how to” do something, your answer should be an activity.
                - Your answer should be in the context of a Teacher engaging with students in a classroom setting
        
        
            Example of 'answer':
            --------------------
            When dealing with behavioral issues in children, it is important to ........
            Source: unmukh-teacher-handbook.pdf,  page# 49
        
        
            Given the following documents:
            ----------------------------
            {contexts}
        """

        system_rules = system_rules.format(contexts=contexts)
        print("\nPROMPT:: ", system_rules)
        print("\nUSER QUERY:: ", user_query)

        os.environ["OPENAI_API_KEY"] = "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"
        os.environ["GPT_MODEL"] = "gpt-4"

        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # client = openai.AzureOpenAI(
        #             azure_endpoint=os.environ["OPENAI_API_BASE"],
        #             api_key=os.environ["OPENAI_API_KEY"],
        #             api_version=os.environ["OPENAI_API_VERSION"]
        #         )
        result = client.chat.completions.create(
            model=os.environ["GPT_MODEL"],
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_query}
            ],
        )

        ragas_ai_model = LangchainLLM(client)

        print("\nRESULT:: ", result)

        message = result.choices[0].message.model_dump()
        response = message["content"]

        data = {
            "query": user_query,
            "result": response,
            "source_documents": source_documents,
            "ground_truths": ground_truths
        }
        # dataset = Dataset.from_dict(data)

        print("\nDATA:: ", data, "\n\n")

        # hf_embeddings = HuggingfaceEmbeddings(model_name="BAAI/bge-small-en")   # size 133MB
        # hf_embeddings = HuggingfaceEmbeddings(model_name="BAAI/bge-large-en-v1.5") # size 1.34GB
        hf_embeddings = HuggingfaceEmbeddings(model_name="flax-sentence-embeddings/all_datasets_v4_mpnet-base")  # size 438MB

        answer_similarity = AnswerSimilarity(llm=ragas_ai_model, embeddings=hf_embeddings)
        answer_correctness = AnswerCorrectness(llm=ragas_ai_model)
        answer_relevancy = AnswerRelevancy(llm=ragas_ai_model, embeddings=hf_embeddings)

        # # create evaluation chains
        faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
        answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
        answer_correctness_chain = RagasEvaluatorChain(metric=answer_correctness)
        answer_similarity_chain = RagasEvaluatorChain(metric=answer_similarity)
        context_rel_chain = RagasEvaluatorChain(metric=context_precision)
        context_precision_chain = RagasEvaluatorChain(metric=context_precision)
        harmfulness_chain = RagasEvaluatorChain(metric=harmfulness)
        context_recall_chain = RagasEvaluatorChain(metric=context_recall)

        # make eval chains
        eval_chains = {
            m.name: RagasEvaluatorChain(metric=m)
            for m in [harmfulness, faithfulness, answer_correctness, context_relevancy, context_recall, context_precision]
        }

        for name, eval_chain in eval_chains.items():
            score_name = f"{name}_score"
            score = eval_chain.invoke(data)[score_name]
            row[score_name] = score

        rows.append(row)

    print("\nROWS:: ", rows)
    writer.writerows(rows)
