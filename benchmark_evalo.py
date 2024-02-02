import os

import marqo
from langchain.chains import RetrievalQA
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.marqo import Marqo
from langchain_community.chat_models import ChatOpenAI
from datasets import Dataset

marqo_url = os.getenv("MARQO_URL", None)
marqoClient = marqo.Client(url=marqo_url)

vectorstore = Marqo(marqoClient, "sakhi_teacher_activities_flaxbase")

marqo_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

relevant_docs = marqo_retriever.get_relevant_documents("Game using two sticks?")

print(relevant_docs)
print(len(relevant_docs))

for doc in relevant_docs:
  print(doc.page_content)
  print('\n')

prompt_template = """You are a simple AI assistant specially programmed to help a teacher with learning and teaching materials for development of children in the age group of 3 to 8 years. Your knowledge base includes only the given documents.
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
    {context}

    QUESTION:```{question}```
    ANSWER:
"""


prompt = ChatPromptTemplate.from_template(prompt_template)


chain_type_kwargs = {"prompt": prompt}

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name='gpt-3.5-turbo-16k',
                                                openai_api_key=os.environ["OPENAI_API_KEY"],
                                                temperature=0),
                                 chain_type="stuff",
                                 chain_type_kwargs={"prompt": prompt},
                                 retriever=marqo_retriever,
                                 return_source_documents=True
                                 )
#
questions = input("Please provide the symptoms here :")
print(questions)
result = qa(questions)
#
print(result.keys())



from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_recall,
    answer_relevancy,
    context_precision,
    context_relevancy,
)
from ragas.metrics.critique import harmfulness

# metrics
metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    harmfulness,
    context_relevancy,
    context_precision,
]

dataset = Dataset.from_dict(data_samples)


result = evaluate(
    dataset,
    metrics=metrics,
)

print(result)