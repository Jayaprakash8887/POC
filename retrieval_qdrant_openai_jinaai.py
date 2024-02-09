import argparse
from typing import (
    Any,
    List,
    Tuple
)

import requests
from qdrant_client import QdrantClient
from langchain.docstore.document import Document


def main():
    parser = argparse.ArgumentParser(description="Q/A using Sakhi documents.")
    parser.add_argument("--question", type=str,
                        help="Query to perform Q/A",
                        default="explain FLN Framework")
    parser.add_argument("--vector_db_host", type=str,
                        help="IP Address of host where the vectordb is located",
                        default="0.0.0.0")
    parser.add_argument("--vector_db_port", type=int,
                        help="Port of Vector DB",
                        default=6333)
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
    answer_qdrant = retrieve_qdrant(args)
    # ai_response = chain.invoke(query)
    # print(ai_response)


def retrieve_qdrant(args):
    collection_name = "teacher_docs"
    client = QdrantClient(host=args.vector_db_host, port=args.vector_db_port)

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

    search_result = client.search(
        collection_name=collection_name, query_vector=embeddings[0], limit=3
    )

    print("search_result:: ", search_result)

    documents = []
    for result in search_result:
        text = result.payload['text']
        print("result.payload.text:: ", text)
        del result.payload['text']
        print("result.payload:: ", result.payload)
        print("result.score:: ", result.score)
        tpl = (Document(page_content=text, metadata=result.payload), result.score)
        documents.append(tpl)

    top_docs_to_fetch = 5
    min_score = 0.7
    print(f"\n\ndocuments : {str(documents)}")

    filtered_document = get_score_filtered_documents(documents, float(min_score))
    print(f"\n\nScore filtered documents : {str(filtered_document)}")
    filtered_document = filtered_document[:int(top_docs_to_fetch)]
    print(f"\n\nTop documents : {str(filtered_document)}")
    contexts = get_formatted_documents(filtered_document)
    print("\n\ncontexts:: ", contexts)

def get_score_filtered_documents(documents: List[Tuple[Document, Any]], min_score=0.0):
    return [(document, search_score) for document, search_score in documents if search_score > min_score]


def get_formatted_documents(documents: List[Tuple[Document, Any]]):
    sources = ""
    for document, _ in documents:
        sources += f"""
            > {document.page_content} \n Source: {document.metadata['file_name']},  page# {document.metadata['page_label']};\n\n
            """
    return sources



if __name__ == "__main__":
    main()