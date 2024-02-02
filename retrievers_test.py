import os

import marqo
from langchain.vectorstores.marqo import Marqo
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from rank_bm25 import BM25Okapi

marqo_url = os.getenv("MARQO_URL", None)
marqoClient = marqo.Client(url=marqo_url)

vectorstore = Marqo(marqoClient, "sakhi_teacher_activities_flaxbase")

marqo_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

user_query = "How to teach about good touch and bad touch?"

relevant_docs = marqo_retriever.get_relevant_documents(user_query)

# print(relevant_docs)
# print(len(relevant_docs))

bm25_retriever = BM25Retriever.from_documents(relevant_docs)

bm25_retriever.k = 20

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, marqo_retriever], weights=[0.42, 0.58])

ensemble_docs = ensemble_retriever.get_relevant_documents(query=user_query)

print("ensemble_docs:: ", ensemble_docs)

documents = [doc.page_content for doc in ensemble_docs]
# Create a BM25 model with default parameters
bm25 = BM25Okapi(documents)

# Rerank documents using BM25 scores
reranked_documents = bm25.get_top_n(user_query, documents, n=len(documents))
print("reranked_documents:: ", reranked_documents)

# Print reranked document IDs and scores
print("Reranked documents using BM25:")
for i, (doc_id, score) in enumerate(reranked_documents):
    print(f"{i+1}. ID: {doc_id}, Score: {score:.4f}")
