import os
os.environ["OPENAI_API_KEY"] = "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"

from datasets import load_dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
print(type(amnesty_qa))
print(amnesty_qa)

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
)

from ragas import evaluate

result = evaluate(
    amnesty_qa["eval"],
    metrics=[
        faithfulness,
        answer_relevancy,
    ],
)

result

df = result.to_pandas()
df.head()