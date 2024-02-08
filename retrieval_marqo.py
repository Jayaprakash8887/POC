import argparse
from typing import (
    Any,
    List,
    Tuple
)

import marqo
from langchain.docstore.document import Document
from langchain.vectorstores.marqo import Marqo


def main():
    parser = argparse.ArgumentParser(description="Q/A using Sakhi documents.")
    parser.add_argument("--question", type=str,
                        help="Query to perform Q/A",
                        default="explain FLN Framework")
    parser.add_argument("--marqo_url", type=str,
                        help="IP Address of host where the vectordb is located",
                        default="0.0.0.0")
    parser.add_argument("--index_name", type=int,
                        help="Port of Vector DB",
                        default="teacher_docs")

    args = parser.parse_args()
    answer_qdrant = retrieve_marqo(args)
    # ai_response = chain.invoke(query)
    # print(ai_response)


def retrieve_marqo(args):
    marqo_client = marqo.Client(url=args.marqo_url)
    retriever = Marqo(marqo_client, args.index_name, searchable_attributes=["text"])

    # Retrieve initial set of documents
    search_results = retriever.similarity_search_with_score(args.question, k=20)
    # print("initial_results:: ", initial_results, "\n\n")

    top_docs_to_fetch = 5
    min_score = 0.7
    print(f"\n\ndocuments : {str(search_results)}")

    filtered_document = get_score_filtered_documents(search_results, float(min_score))
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