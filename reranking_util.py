import os
import marqo
from langchain.vectorstores.marqo import Marqo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from rank_bm25 import BM25Okapi


def retrieve_and_rerank(query):
    # Assuming you have a Marqo instance (make sure to configure Marqo with the correct settings)
    marqo_url = os.getenv("MARQO_URL", None)
    marqo_client = marqo.Client(url=marqo_url)
    retriever = Marqo(marqo_client, "touch_contents_s1", searchable_attributes=["text"])

    # Retrieve initial set of documents
    initial_results = retriever.similarity_search_with_score(query, k=10)
    # print("initial_results:: ", initial_results, "\n\n")

    # Extract passages from the initial results
    document_ids = [doc[0].metadata for doc in initial_results]
    documents = [doc[0].page_content for doc in initial_results]
    marqo_scores = [doc[1] for doc in initial_results]
    print("marqo_scores:: ", marqo_scores, "\n\n")

    # Calculate BM25 scores
    bm25 = BM25Okapi(documents)
    bm25_scores = bm25.get_scores(query)
    print("bm25_scores:: ", bm25_scores, "\n\n")

    pairs = [(query, doc) for doc in documents]
    # print("pairs: ", pairs, "\n\n")

    # # Load and process documents for BERT
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    encoded_inputs = bert_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
    # Perform BERT inference on document-query pairs
    with torch.no_grad():
        outputs = bert_model(**encoded_inputs)
        logits = outputs.logits
        bert_scores = torch.sigmoid(logits[:, 1])  # Assuming positive class represents relevance
        print("bert_scores:: ", bert_scores, "\n\n")

    # # Load and process documents for BAAI
    baai_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
    baai_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")
    with torch.no_grad():
        inputs = baai_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
        baai_scores = baai_model(**inputs, return_dict=True).logits.view(-1, ).float()
        print("baai_scores:: ", baai_scores, "\n\n")

    marqo_sorted_docs = sorted(zip(documents, marqo_scores), key=lambda x: x[1], reverse=True)
    bm25_sorted_docs = sorted(zip(documents, bm25_scores), key=lambda x: x[1], reverse=True)
    bert_sorted_docs = sorted(zip(documents, bert_scores), key=lambda x: x[1], reverse=True)
    baai_sorted_docs = sorted(zip(documents, baai_scores), key=lambda x: x[1], reverse=True)

    print("marqo_sorted_docs:: ", marqo_sorted_docs, "\n\n")
    print("bm25_sorted_docs:: ", bm25_sorted_docs, "\n\n")
    print("bert_sorted_docs:: ", bert_sorted_docs, "\n\n")
    print("baai_sorted_docs:: ", baai_sorted_docs, "\n\n")

    # # Combine scores with weights (adjust weights based on your evaluation)
    marqo_weight = 0.5
    bm25_weight = 0.12
    bert_weight = 0.25
    baai_weight = 0.13
    # # combined_scores = marqo_weight * result_scores + bm25_weight * bm25_scores + bert_weight * bert_scores
    combined_scores = []
    i = 0
    for score in marqo_scores:
        # print("doc:: ", documents[i], " || marqo_score:: ", (marqo_weight * score), " || bm25_score:: ", (bm25_weight * 0.01 * bm25_scores[i]), " || bert_score:: ", (bert_weight * bert_scores[i]), " || baai_scores:: ", (baai_weight * 0.1 * baai_scores[i]), " || TOTAL:: ", (marqo_weight * score + bm25_weight * 0.01 * bm25_scores[i] + bert_weight * bert_scores[i] + baai_weight * 0.1 * baai_scores[i]))
        combined_scores.append(marqo_weight * score + bm25_weight * 0.01 * bm25_scores[i] + bert_weight * bert_scores[i] + baai_weight * 0.1 * baai_scores[i])
        i += 1

    # Sort documents based on combined scores
    sorted_ids = sorted(zip(documents, combined_scores), key=lambda x: x[1], reverse=True)

    # Print reranked document IDs and scores
    print("\n\nReranked documents and scores:")
    for doc, score in sorted_ids:
        print(f"\tID: {doc}, Score: {score:.4f}")


# # Example usage
query = "How to teach about safe touch and unsafe touch?"
print("query:: ", query)
reranked_results = retrieve_and_rerank(query)
