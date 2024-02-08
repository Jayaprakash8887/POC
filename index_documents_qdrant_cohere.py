import argparse
from typing import (
    Dict,
    List
)
import time
import requests
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import SimpleDirectoryReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Batch
from qdrant_client.http.models import Distance, VectorParams


def load_documents(folder_path, input_chunk_size, input_chunk_overlap):
    source_chunks = []
    sources = SimpleDirectoryReader(
        input_dir=folder_path, recursive=True).load_data()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=input_chunk_size, chunk_overlap=input_chunk_overlap)
    for source in sources:
        for chunk in splitter.split_text(source.text):
            source_chunks.append(Document(page_content=chunk, metadata={
                "text": chunk,
                "page_label": source.metadata.get("page_label"),
                "file_name": source.metadata.get("file_name")
            }))
    return source_chunks


def get_formatted_documents(documents: List[Document]):
    docs: List[Dict[str, dict]] = []
    for d in documents:
        doc = {
            "text": d.page_content,
            "metadata": d.metadata if d.metadata else {},
        }
        docs.append(doc)
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qdrant_url',
                        type=str,
                        required=True,
                        help='Endpoint URL of qdrant',
                        )
    parser.add_argument('--index_name',
                        type=str,
                        required=True,
                        help='Name of qdrant index',
                        )
    parser.add_argument('--folder_path',
                        type=str,
                        required=True,
                        help='Path to the folder',
                        default="input_data"
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
    parser.add_argument('--embedding_api_key',
                        type=str,
                        required=True,
                        help='embedding api key'
                        )
    parser.add_argument('--embedding_size',
                        type=int,
                        required=False,
                        help='embedding vector size',
                        default=768
                        )
    parser.add_argument('--chunk_size',
                        type=int,
                        required=False,
                        help='documents chunk size',
                        default=512
                        )
    parser.add_argument('--chunk_overlap',
                        type=int,
                        required=False,
                        help='documents chunk size',
                        default=50
                        )
    parser.add_argument('--fresh_index',
                        action='store_true',
                        help='Is the indexing fresh'
                        )

    args = parser.parse_args()

    QDRANT_URL = args.qdrant_url
    QDRANT_INDEX_NAME = args.index_name
    FOLDER_PATH = args.folder_path
    FRESH_INDEX = args.fresh_index

    EMBEDDING_API_URL = args.embedding_api_url
    EMBEDDING_API_KEY = args.embedding_api_key
    EMBEDDING_MODEL = args.embedding_model
    EMBEDDING_SIZE = args.embedding_size
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    # Initialize Marqo instance
    client = QdrantClient(QDRANT_URL, port=6333)
    if FRESH_INDEX:
        try:
            client.delete_collection(collection_name=QDRANT_INDEX_NAME)
            print("Existing Collection successfully deleted.")
        except:
            print("Collection does not exist. Creating new Collection")

        client.create_collection(
            collection_name=QDRANT_INDEX_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )
        print(f"Collection {QDRANT_INDEX_NAME} created.")

    print("Loading documents...")
    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Total Documents ===>", len(documents))

    f = open("indexed_documents.txt", "w")
    f.write(str(documents))
    f.close()

    print(f"Indexing documents...")
    formatted_documents = get_formatted_documents(documents)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EMBEDDING_API_KEY}",
    }

    qdrant_input = []
    metadata_list = []
    for document in formatted_documents:
        qdrant_input.append(document.get('text'))
        metadata_list.append(document.get('metadata'))

    # !!! _document_batch_size cannot be greater than 96 for Cohere !!!
    _document_batch_size = 50
    computed_docs = 0
    while computed_docs < len(qdrant_input):
        _document_batch_size = _document_batch_size if (len(qdrant_input) - computed_docs) > _document_batch_size else len(qdrant_input) - computed_docs

        data = {
            "texts": qdrant_input[computed_docs:(computed_docs + _document_batch_size)],
            "model": EMBEDDING_MODEL,
            "input_type": 'search_document'
        }

        response = requests.post(EMBEDDING_API_URL, headers=headers, json=data)
        # print("response:: ", response.json())
        embeddings = response.json()["embeddings"]

        client.upsert(
            collection_name=QDRANT_INDEX_NAME,
            points=Batch(
                ids=list(range(computed_docs, (computed_docs + _document_batch_size))),
                vectors=embeddings,
                payloads=metadata_list[computed_docs:(computed_docs + _document_batch_size)]
            ),
        )
        computed_docs += _document_batch_size
        print("computed_docs:: ", computed_docs)
        # below sleep code is introduced to avoid any rate limit issues
        time.sleep(2)

    print("============ INDEX DONE =============", computed_docs)


if __name__ == "__main__":
    main()

# RUN below command for Cohere
# python3 index_documents_qdrant_cohere.py --qdrant_url=http://0.0.0.0:6333 --index_name={COLLECTION_NAME} --embedding_model=embed-english-v3.0 --embedding_api_url=https://api.cohere.ai/v1/embed --embedding_api_key={COHERE_API_KEY} --embedding_size=1024 --folder_path={FOLDER_PATH_OF_DOCS_TO_INDEX} --fresh_index (FOR FRESH INDEXING)
