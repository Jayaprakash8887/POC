import os

import marqo
import openai
from datasets import Dataset
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores.marqo import Marqo

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_recall,
    # answer_relevancy,
    context_precision,
    context_relevancy,
)
from ragas.metrics.critique import harmfulness

# metrics
metrics = [
    faithfulness,
    # answer_relevancy,
    context_recall,
    harmfulness,
    context_relevancy,
    context_precision,
]

# # create evaluation chains
# faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
# answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
# context_rel_chain = RagasEvaluatorChain(metric=context_precision)
# context_precision_chain = RagasEvaluatorChain(metric=context_precision)
# harmfulness_chain = RagasEvaluatorChain(metric=harmfulness)
# context_recall_chain = RagasEvaluatorChain(metric=context_recall)

#
# marqo_url = os.getenv("MARQO_URL", None)
# marqoClient = marqo.Client(url=marqo_url)
#
# vectorstore = Marqo(marqoClient, "sakhi_teacher_activities_flaxbase")
#
# marqo_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
#
#
# prompt_template = """You are a simple AI assistant specially programmed to help a teacher with learning and teaching materials for development of children in the age group of 3 to 8 years. Your knowledge base includes only the given documents.
#     Guidelines:
#         - Always pick relevant 'documents' for the given 'question'. Ensure that your response is directly based on the relevant documents from the given documents.
#         - Your answer must be firmly rooted in the information present in the relevant documents.
#         - Your answer should be in very simple English, for those who may not know English well.
#         - Your answer should not exceed 200 words.
#         - Always return the 'Source' of the relevant documents chosen in the 'answer' at the end.
#         - answer format should strictly follow the format given in the 'Example of answer' section below.
#         - If no relevant document is given, then you should answer "I'm sorry, but I am not currently trained with relevant documents to provide a specific answer for your question.'.
#         - If the question is “how to” do something, your answer should be an activity.
#         - Your answer should be in the context of a Teacher engaging with students in a classroom setting
#
#
#     Example of 'answer':
#     --------------------
#     When dealing with behavioral issues in children, it is important to ........
#     Source: unmukh-teacher-handbook.pdf,  page# 49
#
#
#     Given the following documents:
#     ----------------------------
#     {contexts}
#
# """

# query = "Game using two sticks"
#
# prompt = ChatPromptTemplate.from_template(prompt_template)
# client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# result = client.chat.completions.create(
#     model="gpt-3.5-turbo-16k",
#     messages=[
#         {"role": "system", "content": prompt},
#         {"role": "user", "content": query}
#     ],
# )
#
#
# print(result["result"])
#
# # make eval chains
# eval_chains = {
#     m.name: RagasEvaluatorChain(metric=m)
#     for m in [harmfulness, faithfulness, answer_relevancy, context_relevancy, context_recall]
# }
#
#
# for name, eval_chain in eval_chains.items():
#     score_name = f"{name}_score"
#     print(f"{score_name}: {eval_chain(result)[score_name]}")


# data

data_samples = {
    "question": [
                    "Game using two sticks",
                    "How to teach colours for gifted children"
                ],
    "answer":   [
                    "The game that uses two sticks is called Gilli Danda. It is a game that comes from India. Here's how you play it:\n\nYou need two sticks. The smaller stick is oval and is named Gilli. The longer stick is the Danda. The player uses the Danda to hit the Gilli at one end. This makes the Gilli flip in the air. While it is still in the air, the player must hit Gilli as far as possible. Then the player runs to a point outside a circle before the Gilli is taken by another player. Winning the game is about the best way to raise and hit the Gilli. Any number of players can join this game.\n\nAs well as being fun to play, this game also helps children to develop important skills. These skills include eye-hand coordination and decision making. It can also help with understanding distance and motion.",
                    "You can teach colours to gifted children by involving them in colour mixing activities. You can use different colours of gelatin papers for this activity. Take one coloured paper and place another coloured paper behind it. Allow the children to observe the changes and understand how different colours mix and make a new colour. This activity not only helps them learn about colours but also keeps them interested and engaged because it's a hands-on activity."
                ],
    "contexts": [
                    ["20\nToy-Based Pedagogytheir knees and the remaining 3 players \ntry to avoid being touched by members of \nthe opposing team. It is the next popular \ntag game after kabaddi. Kho-Kho is a traditional Indian sport, which is one of the oldest forms of outdoor sport, dating \nback to prehistoric India. It is most often \nplayed by school children in India and is a competitive game. \nPedagogic Importance: \nFigure 2.21\n \nThis Photo  by Unknown Author is licensed \nPlaying Kho-Kho, \nchildren develop physical stamina and \nthey also learn decision making through \nthis game. \nGilli Danda: Gilli Danda is a thrilling \ngame which originated in India. This game requires two sticks. Method: The smaller stick should be an \noval-shaped wooden piece known as Gilli and the longer stick is known as danda. The player needs to use the danda to hit \nthe Gilli at the raised end, which then flips", "oval-shaped wooden piece known as Gilli and the longer stick is known as danda. The player needs to use the danda to hit \nthe Gilli at the raised end, which then flips \nin the air. When it is in the air, the player needs to hit the Gilli, as far as possible. Then, the player runs to touch a point outside the circle before the Gilli is taken \nby another player. The secret of winning \nthis game lies in how well is the gilli raised and hit. It can be played by any number of players. Pedagogic Importance: This helps in \nenhancing Eye-hand coordination, \ndecision making, estimation and \nmeasurement of distance, and also in learning concepts related to projectile motion, etc.\n2.5 p uppets  \nIn modern times, educationists all over the world have realised the potential of \npuppetry as a medium of communication. \nMany institutions and individuals in India involve students and teachers in the use of puppetry for communicating educational concepts.\nFigure 2.22\nA Puppet is one of the most remarkable"],
                    ["• For example call out, ‘Red’. All the children with red cards will come forward. Similarly, call out \'Green\'. Only the \'greens\' will come forward, and so on.\nVariation Prepare several different colour cards (red, blue, yellow) and give one card to each child. Place one colour card in the centre. Say the name of the card placed in the center (like red) and let them look at their card, identify the colour and match their card with the card placed in the centre. Children having red colour card will come forward and place their card in the centre (matching-matching).\n5 – 6 years • Discuss colours in nature like colour of leaf, flowers, etc., during circle time. Ask the children to draw any scenery like garden, farm, market, etc. Drawing and colouring provides lots of opportunities to experiment with colours.\n• Let the children experiment with colour-mixing to make new shades.\n• Make colour cards of five different shades. Let them arrange these cards from light to dark.", "Annamaya Kosha  and Pranamaya Kosha 89Variation • Find objects of the same colour in the classroom.\n• Let the children match each other’s clothes on the basis of \ncolour.\n• Play circle game with the children holding colour cards for matching.\n• Teach rhymes and songs on colour concept.\n• Play games of classification of colours.\n• Play a game— “tippi-tippi tap what colour you want”.\n• Let the children solve simple riddles on colours.\n• Help children learn and speak out the names of colours.\n• Organise drawing and painting activities as they give children a lot of opportunities to experiment with colours.\n• Threading of coloured beads in a pattern or sequence, for example, red, blue, green, and again red, blue, green and so on is another interesting activity.\nFor Inclusivity: Combine the concepts of texture and colours and \nlet the child sort textured and coloured cards. \nFor Gifted Children:  Colour mixing using different colours using \ngelatin papers of one colour, place second colour paper behind it,"]
                ],
    "ground_truths": [
                        ["Gilli Danda is a thrilling game which originated in India. This game requires two sticks. Method: The smaller stick should be an oval-shaped wooden piece known as Gilli and the longer stick is known as danda. The player needs to use the danda to hit the Gilli at the raised end, which then flips in the air. When it is in the air, the player needs to hit the Gilli, as far as possible. Then, the player runs to touch a point outside the circle before the Gilli is taken by another player. The secret of winning this game lies in how well is the gilli raised and hit. It can be played by any number of players."],
                        ["For Gifted Children: Colour mixing using different colours using gelatin papers of one colour, place second colour paper behind it, a third colour appears."]
                    ]
}
dataset = Dataset.from_dict(data_samples)

# from langchain.chat_models import AzureChatOpenAI
# from langchain.embeddings import AzureOpenAIEmbeddings
# from ragas.llms import LangchainLLM
#
# azure_model = AzureChatOpenAI(
#     deployment_name="My_Jadui_pitara",
#     model="myjp_gpt4",
#     openai_api_base=os.environ["OPENAI_API_BASE"],
#     openai_api_type="azure",
# )
# # wrapper around azure_model
# ragas_azure_model = LangchainLLM(azure_model)
# # patch the new RagasLLM instance
# answer_relevancy.llm = ragas_azure_model
#
# # init and change the embeddings
# # only for answer_relevancy
# azure_embeddings = AzureOpenAIEmbeddings(
#     deployment="your-embeddings-deployment-name",
#     model="your-embeddings-model-name",
#     openai_api_base="https://your-endpoint.openai.azure.com/",
#     openai_api_type="azure",
# )
# # embeddings can be used as it is
# answer_relevancy.embeddings = azure_embeddings
#
# for m in metrics:
#     m.__setattr__("llm", ragas_azure_model)

result = evaluate(
    dataset,
    metrics=metrics,
)

print(result)
