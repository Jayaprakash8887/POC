import argparse
from typing import (
    Any,
    List,
    Tuple
)

import requests
from langchain.docstore.document import Document
from pymilvus import connections, Collection


def main():
    parser = argparse.ArgumentParser(description="Q/A using Sakhi documents.")
    parser.add_argument("--question",
                        type=str,
                        help="Query to perform Q/A",
                        default="conduct an activity using pins and ball")
    parser.add_argument("--vector_db_host",
                        type=str,
                        help="IP Address of host where the vectordb is located",
                        default="0.0.0.0")
    parser.add_argument("--vector_db_port",
                        type=int,
                        help="Port of Vector DB",
                        default=19530)
    parser.add_argument("--vector_db_index",
                        type=str,
                        help="Collection of Vector DB",
                        default="milvus_openai_docs")
    parser.add_argument('--embeddings_api_key',
                        type=str,
                        required=True,
                        help='embedding api key'
                        )
    parser.add_argument('--embedding_model',
                        type=str,
                        required=True,
                        help='data embedding model to be used.'
                        )
    parser.add_argument('--embedding_api_url',
                        type=str,
                        required=True,
                        help='embedding api url'
                        )

    args = parser.parse_args()
    answer_milvus = retrieve(args)
    # ai_response = chain.invoke(query)
    # print(ai_response)


def retrieve(args):
    connections.connect(host=args.vector_db_host, port=args.vector_db_port)
    collection = Collection(args.vector_db_index)  # Get an existing collection.
    collection.load()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.embeddings_api_key}",
    }

    data = {
        "input": args.question,
        "model": args.embedding_model,
    }

    response = requests.post(args.embedding_api_url, headers=headers, json=data)
    embeddings = [d["embedding"] for d in response.json()["data"]]

    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {}
    }

    search_result = collection.search(data=embeddings, anns_field="text_vector", param=search_params, limit=20, expr=None, output_fields=['text', 'metadata'], consistency_level="Strong")

    documents = []
    for result in search_result[0]:
        text = result.fields['text']
        metadata = result.fields['metadata']
        tpl = (Document(page_content=text, metadata=metadata), result.distance)
        documents.append(tpl)

    top_docs_to_fetch = 5
    max_score = 0.5
    print(f"\n\ndocuments : {str(documents)}")

    filtered_document = get_score_filtered_documents(documents, float(max_score))
    print(f"\n\nScore filtered documents : {str(filtered_document)}")
    filtered_document = filtered_document[:int(top_docs_to_fetch)]
    print(f"\n\nTop documents : {str(filtered_document)}")
    contexts = get_formatted_documents(filtered_document)
    print("\n\ncontexts:: ", contexts)


def get_score_filtered_documents(documents: List[Tuple[Document, Any]], max_score=0.0):
    return [(document, search_score) for document, search_score in documents if search_score < max_score]


def get_formatted_documents(documents: List[Tuple[Document, Any]]):
    sources = ""
    for document, _ in documents:
        sources += f"""
            > {document.page_content} \n Source: {document.metadata['file_name']},  page# {document.metadata['page_label']};\n\n
            """
    return sources


if __name__ == "__main__":
    main()
